from src.train.trainer import _has_param_metrics


def test_has_param_metrics_detects_param_prefix():
    assert _has_param_metrics({"avg_total_nll": -1.0, "param_p": 1.2})


def test_has_param_metrics_returns_false_when_no_param_prefix():
    assert not _has_param_metrics({"avg_total_nll": -1.0, "loss": 0.1})
    assert not _has_param_metrics(None)

