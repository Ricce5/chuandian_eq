from src.train import trainer


class _DummyModel:
    def __init__(self):
        self.called = 0

    def print_params(self):
        self.called += 1


def test_log_etas_params_calls_print_params_for_etas():
    model = _DummyModel()
    trainer._maybe_log_etas_params(model, "etas", 3, enabled=True)
    assert model.called == 1


def test_log_etas_params_skips_non_etas_model():
    model = _DummyModel()
    trainer._maybe_log_etas_params(model, "rtpp", 3, enabled=True)
    assert model.called == 0


def test_log_etas_params_respects_disabled_flag():
    model = _DummyModel()
    trainer._maybe_log_etas_params(model, "etas", 3, enabled=False)
    assert model.called == 0


def test_log_etas_params_supports_ema_wrapped_model():
    class _Wrapped:
        def __init__(self, module):
            self.module = module

    model = _DummyModel()
    wrapped = _Wrapped(model)
    trainer._maybe_log_etas_params(wrapped, "etas", 3, enabled=True)
    assert model.called == 1
