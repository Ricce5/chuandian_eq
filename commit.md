修正tpp的arrival_time归一化(减去起始时间)
mamba2Rotary实现

问题
tmp_batch 的pad与现有版本不兼容
新增加数据集序列长度过长
之前的head过深产生问题
全局flash_attn的显存占用小于加入window_size的attn
thp的时间信息没有充分利用，应该加入
建模概率分布的方式不如建模条件强度函数的方式
是否使用mini_batch在训练和测试阶段对单位时间上的nll几乎无影响
weibull分布数值问题
scale rope的center只对数值产生影响，因为不改变相对位置
使用fused_add_norm会导致测试集性能下降
mamba模块使用小卷积核，大核容易过拟合，使得测试集性能不好

mini
debug: thp不除t_max
tpp_trainstep增加梯度裁剪


ps
selective_state_update 中0被替换为(0,0), 处理None下的情况
mamba2 xBC.contiguous().transpose(1, 2), 加入contiguous
加载ckpt中args存在隐患，只能够覆盖，ckpt中没有的参数会保在args中
