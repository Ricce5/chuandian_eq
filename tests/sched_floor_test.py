import torch
import matplotlib.pyplot as plt
from src.train.sched_floor import CosineWithWarmupFloor, LinearWithWarmupFloor

# 假设参数
base_lr = 2e-4
min_lr = 2e-5
warmup_steps = 100
total_steps = 1000

# 虚拟 optimizer
model = torch.nn.Linear(10, 10)
optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

# 测试两种 scheduler
cosine_sched = CosineWithWarmupFloor(optimizer, warmup_steps, total_steps, min_lr=min_lr)
linear_sched = LinearWithWarmupFloor(optimizer, warmup_steps, total_steps, min_lr=min_lr)

# 记录学习率
cosine_lrs, linear_lrs = [], []

for step in range(total_steps):
    optimizer.step()                # 假装更新
    cosine_sched.step()
    linear_sched.step()

    cosine_lrs.append(cosine_sched.get_last_lr()[0])
    linear_lrs.append(linear_sched.get_last_lr()[0])

# 保存图像
plt.figure(figsize=(8, 5))
plt.plot(cosine_lrs, label="Cosine with floor")
plt.plot(linear_lrs, label="Linear with floor")
plt.axhline(min_lr, color="red", linestyle="--", label="min_lr")
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title("Warmup + Floor Scheduler Curves")
plt.legend()
plt.grid(True)
plt.savefig("scheduler_curve.png")
plt.close()
