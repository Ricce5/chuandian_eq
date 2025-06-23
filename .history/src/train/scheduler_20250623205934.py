import torch
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
            # Warm-up阶段：线性增长
            lr = self.base_lr * self.step_count / self.warmup_steps
        else:
            # 线性衰减阶段
            decay_steps = self.total_steps - self.warmup_steps
            progress = (self.step_count - self.warmup_steps) / max(1, decay_steps)
            progress = min(max(progress, 0.0), 1.0)  # 限制在 [0, 1]
            lr = self.base_lr - (self.base_lr - self.min_lr) * progress
        return lr

    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.step_count += 1

class NoOpScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer):
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]