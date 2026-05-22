import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import config_loader
import src.train.config_setup as config_setup
from src.models.builders import ModelBuilder
from src.utils.tpp_experiments import load_tpp_catalog
from src.utils.utils import set_seed


@dataclass
class TensorStats:
    shape: list[int]
    dtype: str
    device: str
    numel: int
    finite_count: int
    nan_count: int
    posinf_count: int
    neginf_count: int
    min: float | None
    max: float | None
    mean: float | None


def tensor_stats(t: torch.Tensor) -> TensorStats:
    x = t.detach()
    numel = int(x.numel())
    if numel == 0:
        return TensorStats(
            shape=list(x.shape),
            dtype=str(x.dtype),
            device=str(x.device),
            numel=0,
            finite_count=0,
            nan_count=0,
            posinf_count=0,
            neginf_count=0,
            min=None,
            max=None,
            mean=None,
        )
    xf = x.float()
    finite = torch.isfinite(xf)
    vals = xf[finite]
    return TensorStats(
        shape=list(x.shape),
        dtype=str(x.dtype),
        device=str(x.device),
        numel=numel,
        finite_count=int(finite.sum().item()),
        nan_count=int(torch.isnan(xf).sum().item()),
        posinf_count=int(torch.isposinf(xf).sum().item()),
        neginf_count=int(torch.isneginf(xf).sum().item()),
        min=float(vals.min().item()) if vals.numel() else None,
        max=float(vals.max().item()) if vals.numel() else None,
        mean=float(vals.mean().item()) if vals.numel() else None,
    )


def state_diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["same_shape"] = list(a.shape) == list(b.shape)
    if a.shape == b.shape:
        af = a.detach().float().cpu()
        bf = b.detach().float().cpu()
        both_finite = torch.isfinite(af) & torch.isfinite(bf)
        out["both_finite_count"] = int(both_finite.sum().item())
        if both_finite.any():
            delta = (af - bf).abs()[both_finite]
            out["abs_diff_max"] = float(delta.max().item())
            out["abs_diff_mean"] = float(delta.mean().item())
        else:
            out["abs_diff_max"] = None
            out["abs_diff_mean"] = None
        out["allclose_atol1e-6"] = bool(torch.allclose(af, bf, atol=1e-6, rtol=1e-5, equal_nan=True))
    else:
        out["both_finite_count"] = None
        out["abs_diff_max"] = None
        out["abs_diff_mean"] = None
        out["allclose_atol1e-6"] = False
    return out


def _get_device(device_arg: str, cuda_id: int) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device(f"cuda:{cuda_id}")
    return torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")


def load_model_main_path(model_name: str, checkpoint_dir: Path, device: torch.device):
    cfg_path = Path(f"config/{model_name}.yaml")
    args = config_loader.load_args_from_yaml(str(cfg_path))
    args.model = args.model.lower()

    checkpoint_path = checkpoint_dir / "best_model_1.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = config_setup.load_args_from_checkpoint(args, checkpoint)
    args.minibatch_training = False
    args.load_specific_parts = None
    args.use_sampler = False
    args.model = args.model.lower()
    if getattr(args, "task_type", None) == "tpp" and getattr(args, "event_drop_prob", 0.0):
        args.event_drop_prob = 0.0

    model_builder = ModelBuilder.by_name(args.model.lower())()
    model = model_builder(args, device)
    model, _, _ = config_setup.load_model_from_checkpoint(model, checkpoint)
    model.eval()
    return model, args


def load_model_forecasting_path(checkpoint_path: Path, device: torch.device, compile_model: bool):
    model, ckpt_args = config_setup.load_and_prepare_model(
        checkpoint_path=checkpoint_path,
        device=device,
        compile=compile_model,
    )
    return model, ckpt_args


def build_past_seq(args_obj, data_root: Path, t_forecast: float, device: torch.device):
    dataset_name = args_obj.dataset
    catalog_ds, registry_name, _ = load_tpp_catalog(
        dataset_name,
        base_dir=data_root / dataset_name,
        catalog_cfg=getattr(args_obj, "catalog_cfg", {}),
    )
    test_seq = catalog_ds.test[0]
    past_seq = test_seq.get_subsequence(0, t_forecast)
    if device.type == "cuda":
        past_seq = past_seq.to(device)
    return past_seq, registry_name


def extract_state(model, past_seq, batch_size: int, max_sample_len: int):
    from mamba_ssm.utils.generation import InferenceParams
    import src

    past_batch = src.data.Batch.from_list([past_seq])
    past_seq_len = len(past_seq)
    max_seqlen = past_seq_len + max_sample_len
    inference_params = InferenceParams(
        max_seqlen=max_seqlen,
        max_batch_size=batch_size,
        key_value_memory_dict=model.base_model.encoder.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=max_seqlen,
        ),
    )
    state_all = model.get_current_state(
        past_batch[:, :-1],
        inference_params=inference_params,
        return_all=True,
    )
    current_state = state_all[:, -1:, :].expand(batch_size, -1, -1)
    return state_all, current_state


def report_model_chain(
    name: str,
    model,
    args_obj,
    data_root: Path,
    t_forecast: float,
    device: torch.device,
    batch_size: int,
    max_sample_len: int,
):
    past_seq, registry = build_past_seq(args_obj, data_root, t_forecast, device)
    state_all, current_state = extract_state(model, past_seq, batch_size=batch_size, max_sample_len=max_sample_len)
    return {
        "name": name,
        "dataset": getattr(args_obj, "dataset", None),
        "registry": registry,
        "past_seq_len": int(len(past_seq)),
        "past_seq_t_end": float(past_seq.t_end),
        "state_all_stats": asdict(tensor_stats(state_all)),
        "current_state_stats": asdict(tensor_stats(current_state)),
        "state_all": state_all,
        "current_state": current_state,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare current_state across main.py and forecasting.py chains")
    parser.add_argument("--model", default="mixer_tpp")
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints/mixer_tpp_20260205-110204"))
    parser.add_argument("--checkpoint_file", type=Path, default=Path("checkpoints/mixer_tpp_20260205-110204/best_model_1.pth"))
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--t_forecast", type=float, default=13523.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--max_sample_len", type=int, default=6000)
    parser.add_argument("--out", type=Path, default=Path("tmp/current_state_compare.json"))
    args = parser.parse_args()

    set_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = _get_device(args.device, args.cuda_id)

    model_main, args_main = load_model_main_path(args.model, args.checkpoint_dir, device)
    model_fore_nocompile, args_fore_nocompile = load_model_forecasting_path(args.checkpoint_file, device, compile_model=False)
    model_fore_compile, args_fore_compile = load_model_forecasting_path(args.checkpoint_file, device, compile_model=True)

    chain_main = report_model_chain(
        "main_test_path",
        model_main,
        args_main,
        args.data_root,
        args.t_forecast,
        device,
        args.batch_size,
        args.max_sample_len,
    )
    chain_fore_nocompile = report_model_chain(
        "forecasting_path_compile_false",
        model_fore_nocompile,
        args_fore_nocompile,
        args.data_root,
        args.t_forecast,
        device,
        args.batch_size,
        args.max_sample_len,
    )
    chain_fore_compile = report_model_chain(
        "forecasting_path_compile_true",
        model_fore_compile,
        args_fore_compile,
        args.data_root,
        args.t_forecast,
        device,
        args.batch_size,
        args.max_sample_len,
    )

    report = {
        "device": str(device),
        "seed": int(args.seed),
        "comparisons": {
            "main_vs_fore_no_compile": state_diff(
                chain_main["current_state"], chain_fore_nocompile["current_state"]
            ),
            "main_vs_fore_compile": state_diff(
                chain_main["current_state"], chain_fore_compile["current_state"]
            ),
            "fore_no_compile_vs_fore_compile": state_diff(
                chain_fore_nocompile["current_state"], chain_fore_compile["current_state"]
            ),
        },
        "chains": {
            "main_test_path": {
                k: v for k, v in chain_main.items() if k not in {"state_all", "current_state"}
            },
            "forecasting_path_compile_false": {
                k: v for k, v in chain_fore_nocompile.items() if k not in {"state_all", "current_state"}
            },
            "forecasting_path_compile_true": {
                k: v for k, v in chain_fore_compile.items() if k not in {"state_all", "current_state"}
            },
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
