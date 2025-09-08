# sched_floor.py
import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineWithWarmupFloor(_LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, min_lr=0.0, last_epoch=-1):
        self.num_warmup_steps = int(num_warmup_steps)
        self.num_training_steps = int(num_training_steps)
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 0)
        if step < self.num_warmup_steps:
            f = step / max(1, self.num_warmup_steps)  # 0→1 线性升温
        else:
            progress = (step - self.num_warmup_steps) / max(1, self.num_training_steps - self.num_warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            f = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1→0 余弦退火

        return [self.min_lr + (base_lr - self.min_lr) * f for base_lr in self.base_lrs]


class LinearWithWarmupFloor(_LRScheduler):
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
            f = 1.0 - progress                                    # 1→0 线性

        return [self.min_lr + (base_lr - self.min_lr) * f for base_lr in self.base_lrs]
