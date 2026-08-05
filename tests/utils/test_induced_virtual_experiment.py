import copy

import numpy as np
import torch

from src.data import Sequence, TppDataset
from src.catalogs.virtual_induced import VirtualInducedETASStandard
from src.models.tpp.etas import (
    ETAS,
    ETAS_EVENT_SOURCE_BACKGROUND,
    ETAS_EVENT_SOURCE_TRIGGERED,
    ETAS_PARENT_INDEX_BACKGROUND,
)
from src.utils.induced_virtual_experiment import (
    compare_etas_components,
    decompose_etas_intensity,
    etas_event_responsibilities,
    moving_block_bootstrap_injection,
    normalize_injection_with_reference,
    save_virtual_induced_catalog,
    simulate_etas_virtual_catalog,
)


class _FixedBackground:
    """Small background double for deterministic ETAS virtual experiments."""

    def __init__(self, event_times=(0.25,), intensity=0.2):
        self.event_times = tuple(event_times)
        self.intensity_value = float(intensity)
        self.cached = None

    def cache_batch(self, time_series, time_series_times, cache_lambda=True):
        self.cached = (time_series.detach().clone(), time_series_times.detach().clone())

    def sample_nhpp_inverse(self, *, B, t0, dt, sample_sequence, mu):
        assert sample_sequence is True
        return [torch.tensor(self.event_times, dtype=torch.float64) for _ in range(B)]

    def intensity(self, batch, t_query=None):
        if t_query is None:
            t_query = batch.arrival_times
        return torch.full_like(t_query, self.intensity_value)


def _model(*, bg_model, k=1e-12, mu=0.1):
    return ETAS(
        omori_p_init=1.2,
        omori_c_init=0.1,
        base_rate_init=max(mu, 1e-12),
        productivity_k_init=k,
        productivity_alpha_init=1.0,
        richter_b=1.0,
        mag_completeness=2.0,
        device=torch.device("cpu"),
        bg_model=bg_model,
        fix_mu=True,
        fixed_mu_value=mu,
        enforce_subcritical=False,
        enforce_p_gt_one=False,
    )


def test_moving_block_bootstrap_preserves_contiguous_blocks_and_fixed_normalization():
    injection = np.array([0.0, 1.0, 2.0, 3.0])
    simulated = moving_block_bootstrap_injection(
        injection,
        target_length=8,
        block_size=2,
        random_state=4,
    )

    assert simulated.shape == (8,)
    assert np.all(simulated >= 0.0)
    assert np.allclose(np.diff(simulated.reshape(-1, 2), axis=1), 1.0)

    normalized = normalize_injection_with_reference(
        np.array([0.0, 5.0, 10.0]),
        {"inj_rate": {"min": 0.0, "max": 20.0}},
    )
    np.testing.assert_allclose(normalized, [0.0, 0.25, 0.5])


def test_etas_sample_can_emit_background_and_triggered_truth(monkeypatch):
    model = _model(bg_model=_FixedBackground(event_times=(0.1,)), k=1.0, mu=0.0)
    original_poisson = np.random.poisson
    calls = 0

    def one_generation_only(lam, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return np.ones_like(lam, dtype=np.int64)
        return np.zeros_like(lam, dtype=np.int64)

    monkeypatch.setattr(np.random, "poisson", one_generation_only)
    try:
        sequence = model.sample(
            batch_size=1,
            duration=1.0,
            t_start=0.0,
            random_state=7,
            max_length=10,
            n_jobs=1,
            return_sequences=True,
            return_event_metadata=True,
        )[0]
    finally:
        monkeypatch.setattr(np.random, "poisson", original_poisson)

    source = sequence.etas_source.detach().cpu().numpy()
    parent = sequence.etas_parent_index.detach().cpu().numpy()
    assert source[0] == ETAS_EVENT_SOURCE_BACKGROUND
    assert parent[0] == ETAS_PARENT_INDEX_BACKGROUND
    assert np.count_nonzero(source == ETAS_EVENT_SOURCE_TRIGGERED) == 1
    triggered_index = int(np.flatnonzero(source == ETAS_EVENT_SOURCE_TRIGGERED)[0])
    assert parent[triggered_index] == 0
    assert sequence.etas_generation[triggered_index].item() == 1


def test_virtual_catalog_save_and_component_comparison(tmp_path):
    model = _model(bg_model=_FixedBackground(), k=1e-12, mu=0.1)
    raw_injection = np.array([0.0, 5.0, 10.0])
    times = np.array([0.0, 0.5, 1.0])
    sequence = simulate_etas_virtual_catalog(
        model,
        raw_injection=raw_injection,
        injection_times=times,
        norm_stats={"inj_rate": {"min": 0.0, "max": 20.0}},
        random_state=9,
        max_length=10,
    )

    assert sequence.time_series.shape == (3, 1)
    torch.testing.assert_close(
        sequence.time_series.squeeze(),
        torch.tensor([0.0, 0.25, 0.5]),
    )
    assert sequence.etas_source.tolist() == [ETAS_EVENT_SOURCE_BACKGROUND]

    parts = decompose_etas_intensity(model, sequence, times)
    np.testing.assert_allclose(parts["constant"], 0.1)
    np.testing.assert_allclose(parts["injection"], 0.2)
    np.testing.assert_allclose(parts["background"], 0.3)
    np.testing.assert_allclose(parts["trigger"], 0.0, atol=1e-10)

    responsibilities = etas_event_responsibilities(model, sequence)
    np.testing.assert_allclose(
        responsibilities["background_probability"] + responsibilities["trigger_probability"],
        1.0,
    )
    assert responsibilities["simulation_source"].tolist() == [ETAS_EVENT_SOURCE_BACKGROUND]

    fitted = copy.deepcopy(model)
    comparison = compare_etas_components(model, fitted, sequence, times)
    assert comparison["metrics"]["background_rmse"] == 0.0
    assert comparison["metrics"]["trigger_rmse"] == 0.0

    paths = save_virtual_induced_catalog(
        tmp_path / "virtual",
        sequence,
        val_start=0.5,
        test_start=0.75,
        metadata={"seed": 9, "mag_completeness": 2.0},
    )
    assert all(path.exists() for path in paths.values())
    reloaded = TppDataset.load_from_disk(paths["full_sequence"])[0]
    assert reloaded.etas_parent_index.tolist() == sequence.etas_parent_index.tolist()
    assert reloaded.raw_time_series.shape == sequence.raw_time_series.shape

    catalog = VirtualInducedETASStandard(root_dir=tmp_path, artifact_dir=paths["metadata"].parent)
    assert catalog.metadata["mag_completeness"] == 2.0
    assert len(catalog.train) == len(catalog.val) == len(catalog.test) == 1
