from omegaconf import OmegaConf

from main import _apply_trial_search_space


class _DummyTrial:
    def suggest_float(self, name, low, high, step=None, log=False):
        del low, high, step, log
        if name == "learning_rate":
            return 3e-4
        raise KeyError(name)

    def suggest_int(self, name, low, high, step=1, log=False):
        del low, high, step, log
        if name == "lstm_num_layers":
            return 2
        raise KeyError(name)

    def suggest_categorical(self, name, choices):
        del choices
        if name == "criterion_cfg.beta":
            return 0.2
        raise KeyError(name)


def test_apply_trial_search_space_supports_nested_keys():
    args = OmegaConf.create(
        {
            "learning_rate": 1e-3,
            "lstm_num_layers": 4,
            "criterion_cfg": {
                "beta": 0.1,
            },
        }
    )
    search_space = {
        "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-3, "log": True},
        "lstm_num_layers": {"type": "int", "low": 1, "high": 4},
        "criterion_cfg.beta": {"type": "categorical", "choices": [0.1, 0.2, 0.3]},
    }
    sampled = _apply_trial_search_space(_DummyTrial(), args, search_space)

    assert args.learning_rate == 3e-4
    assert args.lstm_num_layers == 2
    assert args.criterion_cfg.beta == 0.2
    assert sampled["criterion_cfg.beta"] == 0.2
