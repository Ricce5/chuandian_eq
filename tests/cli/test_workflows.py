from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from src.cli import workflows


class _DummyTrainStep:
    @staticmethod
    def test(**kwargs):
        assert kwargs["val_loader"] is None
        assert kwargs["test_loader"] is None
        return {}, {"nll_train_total": -1.0}


def test_run_test_tpp_handles_missing_num_events_test(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "best_model_1.pth"
    checkpoint = {
        "hyperparameters": {
            "model": "mixer_tpp",
            "task_type": "tpp",
            "dataset": "SCEDC",
            "save_dir": str(tmp_path),
        }
    }
    torch.save(checkpoint, checkpoint_path)

    args_cli = SimpleNamespace(
        ckpt_select="best",
        ckpt_epoch=None,
        trial_index=1,
        save_dir=str(tmp_path),
        no_val_threshold=False,
        threshold=None,
    )
    args = OmegaConf.create(
        {
            "save_dir": str(tmp_path),
            "model": "mixer_tpp",
            "task_type": "tpp",
            "dataset": "SCEDC",
            "use_all_data": True,
        }
    )

    captured = {}

    def _fake_get_model_and_data(runtime_args, base_path, device):
        runtime_args.num_events_train = 11
        runtime_args.num_events_val = 0
        return _DummyTrainStep, None, object(), None, None

    def _fake_setup_config(runtime_args, device, train_dataloader=None, checkpoint=None, restore_weights=True):
        return object(), None, None, None, runtime_args

    def _fake_save_metrics(save_dir, metrics_filename, metrics):
        captured["save_dir"] = save_dir
        captured["metrics_filename"] = metrics_filename
        captured["metrics"] = dict(metrics)

    monkeypatch.setattr(workflows, "get_model_and_data", _fake_get_model_and_data)
    monkeypatch.setattr(workflows.config_setup, "setup_config", _fake_setup_config)
    monkeypatch.setattr(workflows, "save_metrics", _fake_save_metrics)

    workflows.run_test(args_cli, args, torch.device("cpu"))

    assert captured["save_dir"] == str(tmp_path)
    assert captured["metrics_filename"] == "metrics_test_best_1.json"
    assert captured["metrics"]["num_events_train"] == 11
    assert captured["metrics"]["num_events_val"] == 0
    assert captured["metrics"]["num_events_test"] == 0
