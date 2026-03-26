# etas_20260326-110936 训练异常复盘

## 现象
- 训练跑到 1500 轮后，训练集与验证集总损失在后期明显上升。
- 最优验证损失出现在约 epoch 990，之后继续训练未再改善。

## 直接结论
- 后期总损失变大，主要不是 time_nll 变差，而是 bg_kl（背景模型 KL 项）快速增大。
- 即：total_loss 的上升几乎由 KL 项主导。

## 证据（来自 tensorboard 标量）
- Loss/train: min = -5.4527（epoch 1034），last = 339.6187（epoch 1500）
- Loss/val: min = -5.5897（epoch 990），last = 1631.9507（epoch 1500）
- 末轮分项：
	- train: avg_time_nll = 5.0052, avg_bg_kl_nll = 334.6135
	- val: avg_time_nll = -4.5434, avg_bg_kl_nll = 1636.4941

由此可见，后期 total 的上升由 bg_kl_nll 拉动，不是 time_nll 本身恶化。

## 为什么会出现“背景潜变量后验逐步偏离”
背景潜变量后验“逐步偏离”，本质是优化动力学不平衡：
- 数据拟合项（time likelihood）持续推动后验去拟合更细波动；
- KL 约束相对不够强或作用过慢；
- 训练时长较长且无早停，导致偏离持续累积；
- 最终 KL 项接管 total loss。

在实现上，这个机制是明确的：
- ETAS 总损失定义：nll_total = nll_time + bg_kl
- 训练默认优化 total
- 因此只要 bg_kl 上涨，total 必然被抬高。

## 相关配置（本次实验）
- beta_kl: 1e-3
- bg_learning_rate: 5e-4
- learning_rate: 5e-3
- loss_reduction: per_event
- scheduler_type: step_warmup
- epochs: 1500

其中最敏感的是：
- KL 权重与损失缩放（beta_kl + reduction）
- 背景模型学习率（bg_learning_rate）
- 训练轮数与是否早停

## 时间线
- 最后一次 New best validation loss 出现在 epoch 990（-5.5897）。
- 之后持续训练至 1500，KL 项持续增大，造成 train/val total 都显著恶化。

## 建议（下次实验优先级）
1. 加 early stopping（监控 val total），最佳点后尽快停止。
2. 降低 bg_learning_rate（例如从 5e-4 降到 1e-4 或 2e-4）。
3. 适度提高 beta_kl，或做 KL warmup/anneal，避免后验漂移累积。
4. 缩短总 epoch（先在 900-1100 区间内验证）。
5. 训练中同步监控 avg_bg_kl_nll，一旦持续单调上升且 val 无改善，提前终止。

## 一句话总结
这次后期损失变大，不是模型“不会拟合时间项”，而是背景潜变量后验逐步偏离先验导致 KL 爆炸，最终主导总损失。