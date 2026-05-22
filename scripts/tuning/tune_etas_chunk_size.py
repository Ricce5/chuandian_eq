#!/usr/bin/env python3
import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.config_loader import load_args_from_yaml
from main import get_model_and_data
import src.train.config_setup as config_setup
from src.train.tpp_train_step import _nll_out_dict
from src.utils.utils import set_seed


def _run_one_step(base_args, chunk_size: int, device: torch.device) -> dict:
    args = deepcopy(base_args)
    args.etas_query_chunk_size = int(chunk_size)
    args.etas_max_history_events = 0
    set_seed(int(getattr(args, "seed", 0)))

    _, _, train_loader, _, _ = get_model_and_data(args, f"data/{args.dataset}", device)
    model, _, _, _, _ = config_setup.setup_config(
        args,
        device,
        train_dataloader=train_loader,
        checkpoint=None,
        restore_weights=False,
    )
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
    optimizer.zero_grad(set_to_none=True)
    batch = next(iter(train_loader)).to(device)

    torch.cuda.reset_peak_memory_stats(device)
    out = _nll_out_dict(model, batch, nll_kwargs=None)
    loss = out["total"].mean()
    loss.backward()
    optimizer.step()

    peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
    return {
        "ok": True,
        "chunk_size": int(chunk_size),
        "peak_allocated_gib": float(peak),
        "peak_reserved_gib": float(reserved),
        "loss": float(loss.detach().cpu().item()),
    }


def main():
    parser = argparse.ArgumentParser(description="Find largest ETAS query chunk size without CUDA OOM.")
    parser.add_argument("--config", type=str, default="config/etas.yaml")
    parser.add_argument(
        "--candidates",
        type=str,
        default="1024,1536,2048,3072,4096,5120,6144",
        help="Comma-separated candidate chunk sizes to test in order.",
    )
    parser.add_argument("--output", type=str, default="tmp/etas_chunk_tuning.json")
    parser.add_argument("--cuda_id", type=int, default=None, help="Override config cuda_id")
    args_cli = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in current environment.")

    args = load_args_from_yaml(args_cli.config)
    if getattr(args, "model", "").lower() != "etas":
        raise ValueError(f"Config model must be 'etas', got {args.model}")

    if args_cli.cuda_id is not None:
        args.cuda_id = str(args_cli.cuda_id)
    device = torch.device(f"cuda:{args.cuda_id}")

    candidates = [int(x.strip()) for x in args_cli.candidates.split(",") if x.strip()]
    os.makedirs(Path(args_cli.output).parent, exist_ok=True)

    results = []
    best = None
    for chunk in candidates:
        torch.cuda.empty_cache()
        try:
            result = _run_one_step(args, chunk, device)
            results.append(result)
            best = chunk
            print(
                f"[OK] chunk={chunk} peak_alloc={result['peak_allocated_gib']:.2f} GiB "
                f"peak_reserved={result['peak_reserved_gib']:.2f} GiB"
            )
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            msg = str(exc).split("\n")[0]
            results.append({"ok": False, "chunk_size": chunk, "error": msg})
            print(f"[OOM] chunk={chunk} -> {msg}")
            break
        except Exception as exc:
            results.append({"ok": False, "chunk_size": chunk, "error": repr(exc)})
            print(f"[ERR] chunk={chunk} -> {exc}")
            break

    summary = {
        "config": args_cli.config,
        "device": str(device),
        "candidates": candidates,
        "best_safe_chunk_size": best,
        "results": results,
    }
    with open(args_cli.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
