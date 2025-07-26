import numpy as np
import matplotlib.pyplot as plt

# 设置参数
N = 1000  # 粒子数
T = 50    # 时间步长（地震观测次数）
process_noise = 0.1  # 过程噪声（b值的变化噪声）
observation_noise = 0.5  # 观测噪声（震级分布观测噪声）

# 初始b值和观测函数
true_b_value = 1.0  # 真实的b值（假设为1）
h = lambda x: x  # 假设观测函数为b值本身

# 粒子滤波算法
particles = np.random.normal(1.0, 0.1, N)  # 初始化粒子
weights = np.ones(N) / N  # 初始化权重
estimated_b_values = []

for t in range(T):
    # 预测步骤：b值变化
    particles += np.random.normal(0, process_noise, N)
    
    # 模拟观测值：震级分布（观测b值）
    z_k = true_b_value + np.random.normal(0, observation_noise)
    
    # 更新步骤：计算每个粒子的权重
    weights = np.exp(-0.5 * ((z_k - particles) ** 2) / observation_noise ** 2)
    weights /= np.sum(weights)  # 归一化
    
    # 重采样步骤
    indexes = np.random.choice(np.arange(N), size=N, p=weights)
    particles = particles[indexes]
    weights = np.ones(N) / N  # 重采样后，所有粒子的权重均等
    
    # 估计b值
    estimated_b_value = np.sum(weights * particles)
    estimated_b_values.append(estimated_b_value)

# 绘制估计结果
plt.plot(estimated_b_values, label='Estimated b-value')
plt.axhline(true_b_value, color='r', linestyle='--', label='True b-value')
plt.xlabel('Time Step')
plt.ylabel('b-value')
plt.legend()
plt.show()
