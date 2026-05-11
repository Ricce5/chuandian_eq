import torch
import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupLinearDecay:
    def __init__(self, optimizer, base_lr, min_lr, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.min_lr = float(0.0 if min_lr is None else min_lr)
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))
        self.step_count = 0
        self.base_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]

        if self.base_lr > 0.0:
            min_lr_ratio = self.min_lr / self.base_lr
            self.min_lrs = [max(0.0, base * min_lr_ratio) for base in self.base_lrs]
        else:
            self.min_lrs = [self.min_lr for _ in self.base_lrs]

        self.min_lrs = [
            min(base_lr_i, min_lr_i)
            for base_lr_i, min_lr_i in zip(self.base_lrs, self.min_lrs)
        ]

    def get_lr(self):
        return self.get_lrs()[0]

    def get_lrs(self):
        if self.step_count < self.warmup_steps:
            warmup_factor = self.step_count / max(1, self.warmup_steps)
            return [base_lr_i * warmup_factor for base_lr_i in self.base_lrs]

        decay_steps = max(1, self.total_steps - self.warmup_steps)
        progress = (self.step_count - self.warmup_steps) / decay_steps
        progress = min(max(progress, 0.0), 1.0)
        return [
            base_lr_i - (base_lr_i - min_lr_i) * progress
            for base_lr_i, min_lr_i in zip(self.base_lrs, self.min_lrs)
        ]

    def step(self):
        lrs = self.get_lrs()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr
        self.step_count += 1

    def state_dict(self):
        return {
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "step_count": self.step_count,
            "base_lrs": self.base_lrs,
            "min_lrs": self.min_lrs,
        }

    def load_state_dict(self, state_dict):
        self.base_lr = float(state_dict.get("base_lr", self.base_lr))
        self.min_lr = float(state_dict.get("min_lr", self.min_lr))
        self.warmup_steps = int(state_dict.get("warmup_steps", self.warmup_steps))
        self.total_steps = int(state_dict.get("total_steps", self.total_steps))
        self.step_count = int(state_dict.get("step_count", self.step_count))
        self.base_lrs = [float(v) for v in state_dict.get("base_lrs", self.base_lrs)]
        self.min_lrs = [float(v) for v in state_dict.get("min_lrs", self.min_lrs)]

class NoOpScheduler(torch.optim.lr_scheduler._LRScheduler):
    """ No operation learning rate scheduler. """
    def __init__(self, optimizer):
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
    

class CosineWithWarmupFloor(_LRScheduler):
    """ Cosine annealing with linear warmup and a minimum learning rate floor. """
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, min_lr=0.0, last_epoch=-1):
        self.num_warmup_steps = int(num_warmup_steps)
        self.num_training_steps = int(num_training_steps)
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 0)
        if step < self.num_warmup_steps:
            f = step / max(1, self.num_warmup_steps)  # 0→1 linear warmup
        else:
            progress = (step - self.num_warmup_steps) / max(1, self.num_training_steps - self.num_warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            f = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1→0 cosine annealing

        return [self.min_lr + (base_lr - self.min_lr) * f for base_lr in self.base_lrs]


class LinearWithWarmupFloor(_LRScheduler):
    """ Linear decay with linear warmup and a minimum learning rate floor. """
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, min_lr=0.0, last_epoch=-1):
        self.num_warmup_steps = int(num_warmup_steps)
        self.num_training_steps = int(num_training_steps)
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 0)
        if step < self.num_warmup_steps:
            f = step / max(1, self.num_warmup_steps)              # 0→1
        else:
            progress = (step - self.num_warmup_steps) / max(1, self.num_training_steps - self.num_warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            f = 1.0 - progress                                    # 1→0 Linear decay

        return [self.min_lr + (base_lr - self.min_lr) * f for base_lr in self.base_lrs]
