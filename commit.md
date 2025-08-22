



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
Rope使用的时间尺度对结果影响很大, 注意根据时间尺度调整scale_base
SCEDC有效的时间输入会导致过拟合(时间信息编码处理有问题？)
？：bayes软标签是否提升点过程似然
? : 位置编码变基提升尺度范围
卷积层和RoPe时间分辨率耦合，大卷积核不适合适合小时间分辨率
scedc不加卷积效果会很差, 使用2不如4的卷积
SCEDC加载预训练效果变差



PE的周期为2pi-2pi *base, 由于有xPos的存在，不会出现表示的混叠

调试


mini
debug: thp不除t_max
tpp_trainstep增加梯度裁剪


ps
selective_state_update 中0被替换为(0,0), 处理None下的情况
mamba2 xBC.contiguous().transpose(1, 2), 加入contiguous
加载ckpt中args存在隐患，只能够覆盖，ckpt中没有的参数会保在args中



注意8.12前clf tau_mean 取0.026 ,因为dt没有在Mc过滤后计算