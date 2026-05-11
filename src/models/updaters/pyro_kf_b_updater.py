import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch

from src.data.sequence import Sequence
from src.utils.utils import day_offsets_to_np_datetime64, set_xaxis_time_locator
from .base import BValueUpdaterBase

try:
    import pyro
    import pyro.distributions as dist
    import pyro.infer.smcfilter as pyro_smcfilter
    from pyro.infer.smcfilter import SMCFilter
except Exception:  # pragma: no cover - handled at runtime when pyro is missing
    pyro = None
    dist = None
    pyro_smcfilter = None
    SMCFilter = None


LN10 = math.log(10.0)
_PYRO_SYSTEMATIC_SAMPLE_PATCHED = False


def _ensure_device_aware_pyro_systematic_sample() -> None:
    """
    Patch Pyro's `_systematic_sample` to sample randomness on `probs.device`.

    Some Pyro versions call `torch.rand(...)` without device, which defaults to CPU
    and crashes when `probs` lives on CUDA during SMC resampling.
    """
    global _PYRO_SYSTEMATIC_SAMPLE_PATCHED
    if _PYRO_SYSTEMATIC_SAMPLE_PATCHED or pyro_smcfilter is None:
        return

    def _systematic_sample_device_aware(probs: torch.Tensor) -> torch.Tensor:
        batch_shape, size = probs.shape[:-1], probs.size(-1)
        n = probs.cumsum(-1).mul_(size).add_(
            torch.rand(batch_shape + (1,), device=probs.device, dtype=probs.dtype)
        )
        n = n.floor_().clamp_(min=0, max=size).long()
        diff = probs.new_zeros(batch_shape + (size + 1,))
        diff.scatter_add_(-1, n, torch.ones_like(probs))
        index = diff[..., :-1].cumsum(-1).long()
        return index

    pyro_smcfilter._systematic_sample = _systematic_sample_device_aware
    _PYRO_SYSTEMATIC_SAMPLE_PATCHED = True


def _to_tensor(
    value: Union[float, torch.Tensor],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Convert a scalar or tensor to a tensor on the target dtype/device."""
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype, device=device)
    return torch.tensor(float(value), dtype=dtype, device=device)


def _weighted_quantile(
    values: torch.Tensor,
    weights: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    """Compute a single weighted quantile from 1D samples."""
    quantile = float(quantile)
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}.")
    sorted_values, sorted_idx = torch.sort(values)
    sorted_weights = weights[sorted_idx]
    cdf = torch.cumsum(sorted_weights, dim=0)
    cdf = cdf / torch.clamp(cdf[-1], min=1e-12)
    q = torch.tensor(quantile, dtype=values.dtype, device=values.device)
    idx = torch.searchsorted(cdf, q, right=False)
    idx = torch.clamp(idx, max=values.numel() - 1)
    return sorted_values[idx]


def _rw_transition(
    prev_state: torch.Tensor,
    dt: torch.Tensor,
    diffusion_scale: float,
    min_dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return random-walk transition parameters ``(loc, scale)`` for ``log_b``."""
    effective_dt = torch.clamp(dt, min=min_dt)
    sigma_t = torch.as_tensor(
        diffusion_scale,
        dtype=prev_state.dtype,
        device=prev_state.device,
    )
    loc = prev_state
    var = sigma_t.square() * effective_dt
    scale = torch.sqrt(torch.clamp(var, min=1e-12))
    return loc, scale


@dataclass
class _PyroDynamicBModel:
    """State-space model for SMC filtering under random-walk ``log_b`` dynamics."""

    process_scale: float
    prior_log_b_loc: float
    prior_log_b_scale: float
    min_dt: float
    dtype: torch.dtype
    device: torch.device

    def init(
        self,
        state,
        obs_excess: Optional[torch.Tensor],
        has_observation: bool,
    ) -> None:
        """Sample initial latent state and optionally condition on first observation."""
        prior_loc = torch.as_tensor(
            self.prior_log_b_loc,
            dtype=self.dtype,
            device=self.device,
        )
        prior_scale = torch.as_tensor(
            self.prior_log_b_scale,
            dtype=self.dtype,
            device=self.device,
        )
        log_b = pyro.sample(
            "log_b",
            dist.Normal(prior_loc, prior_scale),
        )
        state["log_b"] = log_b

        if has_observation:
            assert obs_excess is not None
            obs_excess = obs_excess.to(dtype=log_b.dtype, device=log_b.device)
            rate = torch.clamp(log_b.exp() * LN10, min=1e-8)
            pyro.sample("obs", dist.Exponential(rate), obs=obs_excess)

    def step(
        self,
        state,
        dt: torch.Tensor,
        obs_excess: Optional[torch.Tensor],
        has_observation: bool,
    ) -> None:
        """Propagate one step and optionally apply the observation likelihood."""
        dt = dt.to(dtype=state["log_b"].dtype, device=state["log_b"].device)
        loc, scale = _rw_transition(
            prev_state=state["log_b"],
            dt=dt,
            diffusion_scale=self.process_scale,
            min_dt=self.min_dt,
        )
        log_b = pyro.sample("log_b", dist.Normal(loc, scale))
        state["log_b"] = log_b

        if has_observation:
            assert obs_excess is not None
            obs_excess = obs_excess.to(dtype=log_b.dtype, device=log_b.device)
            rate = torch.clamp(log_b.exp() * LN10, min=1e-8)
            pyro.sample("obs", dist.Exponential(rate), obs=obs_excess)


@dataclass
class _PyroGuidedProposal:
    """Guided proposal distribution for SMC using Laplace approximation."""

    process_scale: float
    prior_log_b_loc: float
    prior_log_b_scale: float
    min_dt: float
    dtype: torch.dtype
    device: torch.device
    newton_steps: int = 5
    min_scale: float = 1e-4
    max_newton_step: float = 2.0

    def _laplace_proposal(
        self,
        prior_loc: torch.Tensor,
        prior_scale: torch.Tensor,
        obs_excess: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build q(log_b | prior, obs) using a Laplace approximation.

        Posterior log-density up to additive constant:
          -0.5 * ((z - mu)^2 / s^2) + z - ln(10) * x * exp(z)
        where z=log_b, mu=prior_loc, s=prior_scale, x=obs_excess.
        """
        prior_var = torch.clamp(prior_scale.square(), min=1e-10)
        inv_prior_var = 1.0 / prior_var
        c = torch.clamp(obs_excess, min=0.0) * LN10
        z = prior_loc

        for _ in range(self.newton_steps):
            exp_z = torch.exp(torch.clamp(z, min=-50.0, max=50.0))
            grad = -(z - prior_loc) * inv_prior_var + 1.0 - c * exp_z
            hess = -inv_prior_var - c * exp_z
            hess = torch.clamp(hess, max=-1e-8)
            step = torch.clamp(grad / hess, min=-self.max_newton_step, max=self.max_newton_step)
            z = z - step

        exp_z = torch.exp(torch.clamp(z, min=-50.0, max=50.0))
        precision = inv_prior_var + c * exp_z
        proposal_scale = torch.rsqrt(torch.clamp(precision, min=1e-10))
        proposal_scale = torch.clamp(proposal_scale, min=self.min_scale)
        return z, proposal_scale

    def init(
        self,
        state,
        obs_excess: Optional[torch.Tensor],
        has_observation: bool,
    ) -> None:
        """Draw initial proposal sample with observation-aware guidance when available."""
        if has_observation:
            assert obs_excess is not None
            obs_excess = obs_excess.to(dtype=self.dtype, device=self.device)
            prior_loc = torch.as_tensor(
                self.prior_log_b_loc,
                dtype=self.dtype,
                device=self.device,
            )
            prior_scale = torch.as_tensor(
                self.prior_log_b_scale,
                dtype=self.dtype,
                device=self.device,
            )
            q_loc, q_scale = self._laplace_proposal(prior_loc, prior_scale, obs_excess)
            pyro.sample("log_b", dist.Normal(q_loc, q_scale))
            return

        prior_loc = torch.as_tensor(
            self.prior_log_b_loc,
            dtype=self.dtype,
            device=self.device,
        )
        prior_scale = torch.as_tensor(
            self.prior_log_b_scale,
            dtype=self.dtype,
            device=self.device,
        )
        pyro.sample("log_b", dist.Normal(prior_loc, prior_scale))

    def step(
        self,
        state,
        dt: torch.Tensor,
        obs_excess: Optional[torch.Tensor],
        has_observation: bool,
    ) -> None:
        """Draw proposal sample for one time step with optional observation guidance."""
        dt = dt.to(dtype=state["log_b"].dtype, device=state["log_b"].device)
        prior_loc, prior_scale = _rw_transition(
            prev_state=state["log_b"],
            dt=dt,
            diffusion_scale=self.process_scale,
            min_dt=self.min_dt,
        )

        if has_observation:
            assert obs_excess is not None
            obs_excess = obs_excess.to(dtype=prior_loc.dtype, device=prior_loc.device)
            q_loc, q_scale = self._laplace_proposal(prior_loc, prior_scale, obs_excess)
            pyro.sample("log_b", dist.Normal(q_loc, q_scale))
            return

        pyro.sample("log_b", dist.Normal(prior_loc, prior_scale))


@BValueUpdaterBase.register("pyro_kf_dynamic")
@BValueUpdaterBase.register("pyro_kf")
class PyroKFDynamicBUpdater(BValueUpdaterBase):
    """
    KF-like online estimator of time-varying Gutenberg-Richter b-value using Pyro.

    The latent state is ``log(b_t)`` with Gaussian random-walk dynamics:
      ``log(b_t) ~ Normal(log(b_{t-1}), process_scale * sqrt(dt_t))``.

    Observation model for events above completeness threshold ``Mc``:
      ``m_t - Mc ~ Exponential(rate=b_t * ln(10))``.

    Inference uses sequential importance resampling (`pyro.infer.SMCFilter`)
    with a guided proposal (not bootstrap). The proposal incorporates the
    current magnitude likelihood via a Laplace approximation, improving sample
    efficiency for the non-Gaussian observation model.
    """

    def __init__(
        self,
        Mc: float = 3.0,
        process_scale: float = 0.08,
        prior_b_mean: float = 1.0,
        prior_b_std: float = 0.35,
        num_particles: int = 512,
        ess_threshold: float = 0.5,
        credible_mass: float = 0.95,
        min_dt: float = 1e-8,
        mag_key: str = "mag",
        time_key: str = "arrival_times",
        write_back: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        rng_seed: Optional[int] = None,
    ):
        if pyro is None or dist is None or SMCFilter is None:
            raise ImportError(
                "Pyro is required for PyroKFDynamicBUpdater. "
                "Please install `pyro-ppl`."
            )
        _ensure_device_aware_pyro_systematic_sample()

        if process_scale <= 0:
            raise ValueError("process_scale must be positive.")
        if prior_b_mean <= 0 or prior_b_std <= 0:
            raise ValueError("prior_b_mean and prior_b_std must be positive.")
        if not isinstance(num_particles, int) or num_particles <= 1:
            raise ValueError("num_particles must be an integer > 1.")
        if not (0.0 < ess_threshold <= 1.0):
            raise ValueError("ess_threshold must be in (0, 1].")
        if not (0.0 < credible_mass < 1.0):
            raise ValueError("credible_mass must be in (0, 1).")
        if min_dt <= 0:
            raise ValueError("min_dt must be positive.")

        self.Mc = float(Mc)
        self.process_scale = float(process_scale)
        self.prior_b_mean = float(prior_b_mean)
        self.prior_b_std = float(prior_b_std)
        self.num_particles = num_particles
        self.ess_threshold = float(ess_threshold)
        self.credible_mass = float(credible_mass)
        self.min_dt = float(min_dt)
        self.mag_key = mag_key
        self.time_key = time_key
        self.write_back = write_back
        self.dtype = dtype
        self.device = device
        self.rng_seed = rng_seed

        self._smc: Optional[SMCFilter] = None
        self._last_time: Optional[torch.Tensor] = None
        self._initialized = False
        self._target_dtype: Optional[torch.dtype] = None
        self._target_device: Optional[torch.device] = None
        self._last_summary: Optional[Dict[str, Union[float, torch.Tensor]]] = None
        self._batch_stream_states: Optional[list[Dict[str, Any]]] = None

    @property
    def credible_interval(self) -> Tuple[float, float]:
        """Return lower/upper quantiles for the configured credible mass."""
        alpha = (1.0 - self.credible_mass) / 2.0
        return alpha, 1.0 - alpha

    def _reset_state(self) -> None:
        """Clear filter state so the next call starts from prior."""
        self._smc = None
        self._last_time = None
        self._initialized = False
        self._target_dtype = None
        self._target_device = None
        self._last_summary = None
        self._batch_stream_states = None

    def clone_for_batch(self, batch_size: int):
        """Return independent updater clones for per-sequence sampling state."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        return [copy.deepcopy(self) for _ in range(batch_size)]

    def sample_b_value(self, mode: str = "mean") -> torch.Tensor:
        if mode not in {"mean", "sample"}:
            raise ValueError("mode must be one of ['mean', 'sample'].")
        target_dtype = self._target_dtype or self.dtype or torch.get_default_dtype()
        target_device = self._target_device or self.device or torch.device("cpu")
        eps = torch.finfo(target_dtype).eps

        if self._last_summary is None:
            base = torch.tensor(self.prior_b_mean, dtype=target_dtype, device=target_device)
            if mode == "sample":
                base = base + torch.randn_like(base) * float(self.prior_b_std)
            return base.clamp_min(eps)

        b_mean = self._last_summary["b_mean"]
        if isinstance(b_mean, torch.Tensor):
            b = b_mean.to(dtype=target_dtype, device=target_device)
        else:
            b = torch.tensor(float(b_mean), dtype=target_dtype, device=target_device)

        if mode == "sample":
            b_sd = self._last_summary.get("b_sd", None)
            if b_sd is None:
                b = b + torch.randn_like(b) * float(self.prior_b_std)
            else:
                if isinstance(b_sd, torch.Tensor):
                    sd = b_sd.to(dtype=target_dtype, device=target_device)
                else:
                    sd = torch.tensor(float(b_sd), dtype=target_dtype, device=target_device)
                sd = torch.nan_to_num(sd, nan=float(self.prior_b_std), posinf=float(self.prior_b_std))
                sd = sd.clamp_min(eps)
                b = b + torch.randn_like(b) * sd
        return b.clamp_min(eps)

    def _summary_to_python(
        self,
        summary: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        return {
            "b_mean": float(summary["b_mean"].item()),
            "b_sd": float(summary["b_sd"].item()),
            "b_lo": float(summary["b_lo"].item()),
            "b_hi": float(summary["b_hi"].item()),
            "ess": float(summary["ess"].item()),
            "used_mask": float(summary["used_mask"].item()),
        }

    def _init_target_dtype_device_for_update(
        self,
        mag: Union[float, torch.Tensor],
        time: Union[float, torch.Tensor],
    ) -> None:
        if self._target_dtype is None:
            self._target_dtype = self.dtype if self.dtype is not None else torch.get_default_dtype()
        if self._target_device is None:
            if isinstance(mag, torch.Tensor):
                self._target_device = mag.device
            elif isinstance(time, torch.Tensor):
                self._target_device = time.device
            else:
                self._target_device = torch.device("cpu")

    def _build_filter(self, dtype: torch.dtype, device: torch.device) -> SMCFilter:
        """Instantiate the Pyro SMC filter with model and guided proposal."""
        prior_mean_t = torch.tensor(self.prior_b_mean, dtype=dtype, device=device)
        prior_std_t = torch.tensor(self.prior_b_std, dtype=dtype, device=device)

        prior_var = prior_std_t.square()
        prior_loc = torch.log(
            prior_mean_t.square() / torch.sqrt(prior_var + prior_mean_t.square())
        )
        prior_scale = torch.sqrt(
            torch.log1p(prior_var / torch.clamp(prior_mean_t.square(), min=1e-12))
        )

        model = _PyroDynamicBModel(
            process_scale=self.process_scale,
            prior_log_b_loc=float(prior_loc.item()),
            prior_log_b_scale=float(prior_scale.item()),
            min_dt=self.min_dt,
            dtype=dtype,
            device=device,
        )
        guide = _PyroGuidedProposal(
            process_scale=self.process_scale,
            prior_log_b_loc=float(prior_loc.item()),
            prior_log_b_scale=float(prior_scale.item()),
            min_dt=self.min_dt,
            dtype=dtype,
            device=device,
        )
        smc = SMCFilter(
            model=model,
            guide=guide,
            num_particles=self.num_particles,
            max_plate_nesting=0,
            ess_threshold=self.ess_threshold,
        )
        smc.state._log_weights = smc.state._log_weights.to(dtype=dtype, device=device)
        return smc

    def _posterior_summary(self) -> Dict[str, torch.Tensor]:
        """Compute posterior mean, spread, interval, and ESS from particles."""
        assert self._smc is not None
        log_b_particles = self._smc.state["log_b"]
        log_weights = self._smc.state._log_weights
        weights = torch.softmax(log_weights, dim=0)
        b_particles = torch.exp(log_b_particles)

        b_mean = torch.sum(weights * b_particles)
        centered = b_particles - b_mean
        b_var = torch.sum(weights * centered * centered)
        b_sd = torch.sqrt(torch.clamp(b_var, min=0.0))
        q_lo, q_hi = self.credible_interval
        b_lo = _weighted_quantile(b_particles, weights, q_lo)
        b_hi = _weighted_quantile(b_particles, weights, q_hi)
        ess = 1.0 / torch.clamp(torch.sum(weights * weights), min=1e-12)
        return {
            "b_mean": b_mean,
            "b_sd": b_sd,
            "b_lo": b_lo,
            "b_hi": b_hi,
            "ess": ess,
        }

    def _update_filter_once(
        self,
        mag: torch.Tensor,
        time: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run one predict-update cycle for a single event."""
        if self._smc is None:
            self._smc = self._build_filter(dtype=mag.dtype, device=mag.device)

        has_observation = bool((mag >= self.Mc).item())
        obs_excess = mag - self.Mc if has_observation else None

        if not self._initialized:
            self._smc.init(obs_excess=obs_excess, has_observation=has_observation)
            self._initialized = True
        else:
            assert self._last_time is not None
            dt = torch.clamp(time - self._last_time, min=0.0)
            self._smc.step(
                dt=dt,
                obs_excess=obs_excess,
                has_observation=has_observation,
            )

        self._last_time = time
        out = self._posterior_summary()
        out["used_mask"] = torch.tensor(
            int(has_observation), dtype=mag.dtype, device=mag.device
        )
        return out

    def _allocate_outputs(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        prefix: str,
    ) -> Dict[str, torch.Tensor]:
        """Allocate output tensors for posterior summaries."""
        outputs = {
            f"{prefix}b_mean": torch.full(shape, float("nan"), dtype=dtype, device=device),
            f"{prefix}b_sd": torch.full(shape, float("nan"), dtype=dtype, device=device),
            f"{prefix}b_lo": torch.full(shape, float("nan"), dtype=dtype, device=device),
            f"{prefix}b_hi": torch.full(shape, float("nan"), dtype=dtype, device=device),
            f"{prefix}ess": torch.full(shape, float("nan"), dtype=dtype, device=device),
            f"{prefix}used_mask": torch.zeros(shape, dtype=torch.int64, device=device),
        }
        return outputs

    def _run_single_sequence_filter(
        self,
        mags: torch.Tensor,
        times: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run filtering for a single 1D sequence and return per-event summaries."""
        if mags.ndim != 1 or times.ndim != 1:
            raise ValueError("mags/times for single-sequence filtering must be 1D.")
        if mags.shape[0] != times.shape[0]:
            raise ValueError("Magnitude and time arrays must share the same length.")
        if mags.shape[0] == 0:
            raise ValueError("Input sequence contains zero events.")
        if torch.any(times[1:] < times[:-1]):
            raise ValueError("Event times must be sorted in non-decreasing order.")

        outputs = self._allocate_outputs(
            shape=(mags.shape[0],),
            dtype=mags.dtype,
            device=mags.device,
            prefix="",
        )

        self._reset_state()
        self._target_dtype = mags.dtype
        self._target_device = mags.device

        for idx in range(mags.shape[0]):
            summary = self._update_filter_once(mag=mags[idx], time=times[idx])
            outputs["b_mean"][idx] = summary["b_mean"]
            outputs["b_sd"][idx] = summary["b_sd"]
            outputs["b_lo"][idx] = summary["b_lo"]
            outputs["b_hi"][idx] = summary["b_hi"]
            outputs["ess"][idx] = summary["ess"]
            outputs["used_mask"][idx] = int(summary["used_mask"].item())

        return outputs

    def _infer_batch_valid_mask(
        self,
        seq,
        mags: torch.Tensor,
    ) -> torch.Tensor:
        """
        Infer valid-event mask for batched inputs.

        Priority:
          1) ``input_mask`` (if present and same shape as mags)
          2) ``end_idx``   (if present and shape [batch])
          3) all positions are treated as valid events
        """
        if "input_mask" in seq:
            input_mask = seq["input_mask"]
            if not isinstance(input_mask, torch.Tensor):
                raise ValueError("input_mask must be a torch.Tensor when provided.")
            if input_mask.shape != mags.shape:
                raise ValueError(
                    "input_mask must match the shape of batched magnitudes/times. "
                    f"Got input_mask={tuple(input_mask.shape)} vs mags={tuple(mags.shape)}."
                )
            return input_mask.to(device=mags.device) > 0

        if "end_idx" in seq:
            end_idx = seq["end_idx"]
            if not isinstance(end_idx, torch.Tensor):
                raise ValueError("end_idx must be a torch.Tensor when provided.")
            if end_idx.ndim != 1 or end_idx.shape[0] != mags.shape[0]:
                raise ValueError(
                    "end_idx must have shape [batch_size] for batched input. "
                    f"Got {tuple(end_idx.shape)} for batch_size={mags.shape[0]}."
                )
            end_idx = end_idx.to(device=mags.device, dtype=torch.long)
            arange = torch.arange(mags.shape[1], device=mags.device).unsqueeze(0)
            return arange < end_idx.unsqueeze(1)

        return torch.ones_like(mags, dtype=torch.bool, device=mags.device)

    def fit(self, seq: Sequence, prefix: str = "") -> Dict[str, torch.Tensor]:
        """
        Run sequential filtering for a full sequence.

        Returns tensors of posterior statistics at each event and optionally
        writes them back into ``seq`` with ``prefix``.
        """
        if self.rng_seed is not None:
            pyro.set_rng_seed(self.rng_seed)

        if self.mag_key not in seq:
            raise KeyError(
                f"Sequence is missing magnitude key '{self.mag_key}'. "
                f"Available keys: {list(seq.keys())}"
            )
        if self.time_key not in seq:
            raise KeyError(
                f"Sequence is missing time key '{self.time_key}'. "
                f"Available keys: {list(seq.keys())}"
            )

        mags = seq[self.mag_key]
        times = seq[self.time_key]
        target_dtype = self.dtype if self.dtype is not None else torch.get_default_dtype()
        mags = mags.to(dtype=target_dtype)
        times = times.to(dtype=target_dtype)
        if self.device is not None:
            mags = mags.to(self.device)
            times = times.to(self.device)

        if mags.shape != times.shape:
            raise ValueError("Magnitude and time arrays must share the same shape.")
        if mags.ndim not in (1, 2):
            raise ValueError(
                "Only 1D [num_events] or 2D [batch_size, seq_len] tensors are supported."
            )

        last_summary: Optional[Dict[str, float]] = None

        if mags.ndim == 1:
            single_outputs = self._run_single_sequence_filter(mags=mags, times=times)
            outputs = {
                f"{prefix}b_mean": single_outputs["b_mean"],
                f"{prefix}b_sd": single_outputs["b_sd"],
                f"{prefix}b_lo": single_outputs["b_lo"],
                f"{prefix}b_hi": single_outputs["b_hi"],
                f"{prefix}ess": single_outputs["ess"],
                f"{prefix}used_mask": single_outputs["used_mask"],
            }
            last_summary = self._summary_to_python(
                {
                    "b_mean": outputs[f"{prefix}b_mean"][-1],
                    "b_sd": outputs[f"{prefix}b_sd"][-1],
                    "b_lo": outputs[f"{prefix}b_lo"][-1],
                    "b_hi": outputs[f"{prefix}b_hi"][-1],
                    "ess": outputs[f"{prefix}ess"][-1],
                    "used_mask": outputs[f"{prefix}used_mask"][-1].to(dtype=target_dtype),
                }
            )
        else:
            batch_size, seq_len = mags.shape
            outputs = self._allocate_outputs(
                shape=(batch_size, seq_len),
                dtype=target_dtype,
                device=mags.device,
                prefix=prefix,
            )
            valid_mask = self._infer_batch_valid_mask(seq=seq, mags=mags)
            if valid_mask.shape != mags.shape:
                raise ValueError(
                    f"Resolved valid mask has invalid shape {tuple(valid_mask.shape)} "
                    f"(expected {tuple(mags.shape)})."
                )

            for b in range(batch_size):
                seq_mask = valid_mask[b]
                valid_indices = torch.nonzero(seq_mask, as_tuple=False).squeeze(-1)
                if valid_indices.numel() == 0:
                    continue

                mags_b = mags[b, valid_indices]
                times_b = times[b, valid_indices]
                single_outputs = self._run_single_sequence_filter(mags=mags_b, times=times_b)

                outputs[f"{prefix}b_mean"][b, valid_indices] = single_outputs["b_mean"]
                outputs[f"{prefix}b_sd"][b, valid_indices] = single_outputs["b_sd"]
                outputs[f"{prefix}b_lo"][b, valid_indices] = single_outputs["b_lo"]
                outputs[f"{prefix}b_hi"][b, valid_indices] = single_outputs["b_hi"]
                outputs[f"{prefix}ess"][b, valid_indices] = single_outputs["ess"]
                outputs[f"{prefix}used_mask"][b, valid_indices] = single_outputs["used_mask"]

                last_summary = self._summary_to_python(
                    {
                        "b_mean": single_outputs["b_mean"][-1],
                        "b_sd": single_outputs["b_sd"][-1],
                        "b_lo": single_outputs["b_lo"][-1],
                        "b_hi": single_outputs["b_hi"][-1],
                        "ess": single_outputs["ess"][-1],
                        "used_mask": single_outputs["used_mask"][-1].to(dtype=target_dtype),
                    }
                )

        if self.write_back:
            for key, value in outputs.items():
                seq[key] = value

        if last_summary is None:
            self._last_summary = {
                "b_mean": float("nan"),
                "b_sd": float("nan"),
                "b_lo": float("nan"),
                "b_hi": float("nan"),
                "ess": float("nan"),
                "used_mask": 0.0,
            }
        else:
            self._last_summary = last_summary
        return outputs

    def update_one(
        self,
        mag: Union[float, torch.Tensor],
        time: Union[float, torch.Tensor],
        active_mask: Optional[Union[bool, torch.Tensor]] = None,
    ) -> Dict[str, Union[float, torch.Tensor]]:
        """
        Update filter state with one event.

        Supported shapes:
          - scalar ``mag`` and scalar ``time``: single-sequence streaming
          - vector ``mag`` and vector ``time`` with shape ``[batch]``:
            batched streaming update (independent filter per batch entry)
        """
        self._init_target_dtype_device_for_update(mag=mag, time=time)
        assert self._target_dtype is not None
        assert self._target_device is not None

        mag_t = _to_tensor(mag, dtype=self._target_dtype, device=self._target_device)
        time_t = _to_tensor(time, dtype=self._target_dtype, device=self._target_device)
        if mag_t.shape != time_t.shape:
            raise ValueError("mag and time must have identical shape in update_one().")

        if active_mask is None:
            active_mask_t = torch.ones_like(mag_t, dtype=torch.bool, device=mag_t.device)
        else:
            active_mask_t = (
                active_mask if isinstance(active_mask, torch.Tensor) else torch.tensor(active_mask)
            )
            active_mask_t = active_mask_t.to(device=mag_t.device)
            if active_mask_t.shape != mag_t.shape:
                raise ValueError(
                    "active_mask shape mismatch in update_one. "
                    f"Got active_mask={tuple(active_mask_t.shape)} vs mag={tuple(mag_t.shape)}."
                )
            active_mask_t = active_mask_t.to(dtype=torch.bool)

        if mag_t.ndim == 0:
            if not bool(active_mask_t.item()):
                if self._last_summary is None:
                    out = {
                        "b_mean": float("nan"),
                        "b_sd": float("nan"),
                        "b_lo": float("nan"),
                        "b_hi": float("nan"),
                        "ess": float("nan"),
                        "used_mask": 0.0,
                    }
                else:
                    out = dict(self._last_summary)
                    out["used_mask"] = 0.0
                self._last_summary = out
                return dict(out)
            if self._batch_stream_states is not None:
                self._reset_state()
                self._init_target_dtype_device_for_update(mag=mag_t, time=time_t)
            if self.rng_seed is not None and not self._initialized:
                pyro.set_rng_seed(self.rng_seed)

            if self._last_time is not None and bool((time_t < self._last_time).item()):
                raise ValueError("Incoming event time must be non-decreasing.")

            summary = self._update_filter_once(mag=mag_t, time=time_t)
            self._last_summary = self._summary_to_python(summary)
            return self._last_summary.copy()

        if mag_t.ndim != 1:
            raise ValueError(
                "update_one supports only scalar () or batched vector (B,) mag/time."
            )

        batch_size = int(mag_t.shape[0])
        if batch_size <= 0:
            raise ValueError("Batched update_one requires a non-empty batch.")

        if self._batch_stream_states is None:
            if self._initialized:
                self._reset_state()
                self._init_target_dtype_device_for_update(mag=mag_t, time=time_t)
            if self.rng_seed is not None:
                pyro.set_rng_seed(self.rng_seed)
            self._batch_stream_states = [
                {
                    "smc": None,
                    "last_time": None,
                    "initialized": False,
                    "last_summary": None,
                }
                for _ in range(batch_size)
            ]
        elif len(self._batch_stream_states) != batch_size:
            raise ValueError(
                "Batched update_one size mismatch with existing state. "
                f"Expected batch={len(self._batch_stream_states)}, got batch={batch_size}."
            )

        b_mean = torch.empty(batch_size, dtype=mag_t.dtype, device=mag_t.device)
        b_sd = torch.empty(batch_size, dtype=mag_t.dtype, device=mag_t.device)
        b_lo = torch.empty(batch_size, dtype=mag_t.dtype, device=mag_t.device)
        b_hi = torch.empty(batch_size, dtype=mag_t.dtype, device=mag_t.device)
        ess = torch.empty(batch_size, dtype=mag_t.dtype, device=mag_t.device)
        used_mask = torch.empty(batch_size, dtype=torch.int64, device=mag_t.device)

        for idx in range(batch_size):
            state = self._batch_stream_states[idx]
            last_time = state["last_time"]
            if (
                bool(active_mask_t[idx].item())
                and last_time is not None
                and bool((time_t[idx] < last_time).item())
            ):
                raise ValueError(
                    f"Incoming event time must be non-decreasing for batch index {idx}."
                )

            if not bool(active_mask_t[idx].item()):
                prev = state["last_summary"]
                if prev is None:
                    b_mean[idx] = float("nan")
                    b_sd[idx] = float("nan")
                    b_lo[idx] = float("nan")
                    b_hi[idx] = float("nan")
                    ess[idx] = float("nan")
                    used_mask[idx] = 0
                else:
                    b_mean[idx] = float(prev["b_mean"])
                    b_sd[idx] = float(prev["b_sd"])
                    b_lo[idx] = float(prev["b_lo"])
                    b_hi[idx] = float(prev["b_hi"])
                    ess[idx] = float(prev["ess"])
                    used_mask[idx] = 0
                continue

            self._smc = state["smc"]
            self._last_time = state["last_time"]
            self._initialized = bool(state["initialized"])
            summary = self._update_filter_once(mag=mag_t[idx], time=time_t[idx])
            state["smc"] = self._smc
            state["last_time"] = self._last_time
            state["initialized"] = self._initialized
            state["last_summary"] = self._summary_to_python(summary)

            b_mean[idx] = summary["b_mean"]
            b_sd[idx] = summary["b_sd"]
            b_lo[idx] = summary["b_lo"]
            b_hi[idx] = summary["b_hi"]
            ess[idx] = summary["ess"]
            used_mask[idx] = int(summary["used_mask"].item())

        self._smc = None
        self._last_time = None
        self._initialized = False

        out = {
            "b_mean": b_mean,
            "b_sd": b_sd,
            "b_lo": b_lo,
            "b_hi": b_hi,
            "ess": ess,
            "used_mask": used_mask,
        }
        self._last_summary = out
        return out

    @staticmethod
    def plot(
        seq: Sequence,
        prefix: str = "",
        field_mean: str = "b_mean",
        field_lo: str = "b_lo",
        field_hi: str = "b_hi",
        field_ess: str = "ess",
        x_axis: str = "event",
        time_key: str = "arrival_times",
        title: str = "Particle-filter dynamic b-value",
        label: str = "Particle filter mean b",
        ax: Optional[plt.Axes] = None,
        show_interval: bool = True,
        show_ess: bool = False,
        start_time=None,
        show: bool = True,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot posterior b-curve from a sequence populated by ``fit()``."""
        mean_key = prefix + field_mean
        lo_key = prefix + field_lo
        hi_key = prefix + field_hi
        ess_key = prefix + field_ess

        if mean_key not in seq:
            raise KeyError(
                f"Sequence does not contain '{mean_key}'. "
                "Please run fit() before plotting."
            )
        if show_interval and (lo_key not in seq or hi_key not in seq):
            raise KeyError(
                f"Sequence does not contain '{lo_key}' or '{hi_key}'. "
                "Please run fit() before plotting."
            )
        if show_ess and ess_key not in seq:
            raise KeyError(
                f"Sequence does not contain '{ess_key}'. "
                "Please run fit() before plotting with ESS."
            )

        b_mean = seq[mean_key].detach().cpu().numpy()

        if x_axis == "event":
            xs = range(len(b_mean))
            xlabel = "Event index"
        elif x_axis == "time":
            if time_key not in seq:
                raise KeyError(f"Sequence is missing the time field '{time_key}'")
            days = seq[time_key].detach().cpu().numpy()
            if start_time is None:
                xs = days
                xlabel = "Time (days)"
            else:
                xs = day_offsets_to_np_datetime64(start_time, days)
                xlabel = "Year"
        else:
            raise ValueError("x_axis must be either 'event' or 'time'")

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 4.6))
        else:
            fig = ax.figure

        ax.plot(xs, b_mean, label=label)

        if show_interval:
            b_lo = seq[lo_key].detach().cpu().numpy()
            b_hi = seq[hi_key].detach().cpu().numpy()
            ax.fill_between(xs, b_lo, b_hi, alpha=0.25, label="Credible interval")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("b-value")
        ax.set_title(title)
        ax.legend(loc="best")

        if show_ess:
            ess = seq[ess_key].detach().cpu().numpy()
            ax2 = ax.twinx()
            ax2.plot(
                xs,
                ess,
                linestyle="--",
                alpha=0.4,
                color="tab:gray",
                label="ESS",
            )
            ax2.set_ylabel("Effective sample size")
            ax2.legend(loc="upper right")

        if x_axis == "time" and start_time is not None:
            set_xaxis_time_locator(ax, start_time)

        if show:
            plt.tight_layout()
            plt.show()

        return fig, ax
