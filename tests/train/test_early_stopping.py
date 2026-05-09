from src.train.early_stopping import EarlyStopping


def test_early_stopping_tracks_best_and_patience():
    early_stopping = EarlyStopping(patience=2, min_delta=0.0, mode="min")

    assert not early_stopping.step(1.0, 1)
    assert early_stopping.state.best_value == 1.0
    assert early_stopping.state.best_epoch == 1

    assert not early_stopping.step(1.1, 2)
    assert early_stopping.step(1.2, 3)


def test_early_stopping_resets_on_improvement():
    early_stopping = EarlyStopping(patience=1, min_delta=0.0, mode="min")

    assert not early_stopping.step(2.0, 1)
    assert early_stopping.step(2.1, 2)

    early_stopping = EarlyStopping(patience=1, min_delta=0.0, mode="min")
    assert not early_stopping.step(2.0, 1)
    assert not early_stopping.step(1.5, 2)
    assert early_stopping.state.best_value == 1.5
    assert early_stopping.state.bad_epochs == 0


def test_early_stopping_min_delta_requires_sufficient_improvement():
    early_stopping = EarlyStopping(patience=2, min_delta=0.1, mode="min")

    assert not early_stopping.step(1.0, 1)
    assert not early_stopping.step(0.95, 2)
    assert early_stopping.state.bad_epochs == 1
    assert not early_stopping.step(0.89, 3)
    assert early_stopping.state.best_value == 0.89
    assert early_stopping.state.bad_epochs == 0


def test_early_stopping_state_roundtrip():
    early_stopping = EarlyStopping(patience=3, min_delta=0.05, mode="min")
    early_stopping.step(1.2, 1)
    early_stopping.step(1.25, 2)

    state = early_stopping.state_dict()

    restored = EarlyStopping(patience=1, min_delta=0.0, mode="min")
    restored.load_state_dict(state)

    assert restored.patience == 3
    assert restored.min_delta == 0.05
    assert restored.state.best_value == 1.2
    assert restored.state.best_epoch == 1
    assert restored.state.bad_epochs == 1
