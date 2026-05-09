from dataclasses import dataclass


@dataclass
class EarlyStoppingState:
    best_value: float = float("inf")
    best_epoch: int = -1
    bad_epochs: int = 0


class EarlyStopping:
    def __init__(self, patience: int | None = None, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = float(min_delta)
        self.mode = mode

        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")

        self.state = EarlyStoppingState(
            best_value=float("inf") if mode == "min" else float("-inf"),
        )

    @property
    def enabled(self):
        return self.patience is not None and self.patience >= 0

    def state_dict(self) -> dict:
        return {
            "best_value": self.state.best_value,
            "best_epoch": self.state.best_epoch,
            "bad_epochs": self.state.bad_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if not state_dict:
            return
        self.state.best_value = state_dict.get("best_value", self.state.best_value)
        self.state.best_epoch = state_dict.get("best_epoch", self.state.best_epoch)
        self.state.bad_epochs = state_dict.get("bad_epochs", self.state.bad_epochs)
        self.patience = state_dict.get("patience", self.patience)
        self.min_delta = state_dict.get("min_delta", self.min_delta)
        self.mode = state_dict.get("mode", self.mode)

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "min":
            return value < (self.state.best_value - self.min_delta)
        return value > (self.state.best_value + self.min_delta)

    def step(self, value: float, epoch: int) -> bool:
        if not self.enabled:
            return False

        if self._is_improvement(value):
            self.state.best_value = value
            self.state.best_epoch = epoch
            self.state.bad_epochs = 0
            return False

        self.state.bad_epochs += 1
        return self.state.bad_epochs >= self.patience
