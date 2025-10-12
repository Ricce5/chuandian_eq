import torch
import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupLinearDecay:
    def __init__(self, optimizer, base_lr, min_lr, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.step_count = 0

    def get_lr(self):
        if self.step_count < self.warmup_steps:
            lr = self.base_lr * self.step_count / self.warmup_steps
        else:
            decay_steps = self.total_steps - self.warmup_steps
            progress = (self.step_count - self.warmup_steps) / max(1, decay_steps)
            progress = min(max(progress, 0.0), 1.0) 
            lr = self.base_lr - (self.base_lr - self.min_lr) * progress
        return lr

    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.step_count += 1

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