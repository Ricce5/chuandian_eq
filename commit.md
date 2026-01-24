去掉clf和reg的震级归一化
使用单位指数归一化input_adpter



selective_state_update 中0被替换为(0,0), 处理None下的情况
mamba2 xBC.contiguous().transpose(1, 2), 加入contiguous


debug:
1. 替换rtpp采样
2. pnr数据重新生成
3. 对bg模型的nll计算进行修正
4. 对bg模型的lambda进行cache
5. 修改get_subsequence, 
6. 修改rtpp的last_surv_time处理逻辑
7. 修改bg_model的 intensity_integral没有clamp_min(0.0)
8. 修改pnr完备性-1.8
9. 注水序列长度54000超过RNN处理范围
10. pnr的generate中转datetime
11. 修改pnr预处理的add_datetime_column
12. 修正M-test错误
13. 修改etas转double的位置
14. bg_model限制fp32
15. 修复bg_model的积分形状问题
16. bg_model采样时加入mu
17. bg_model学习率支持与主模型不同
18. 修改ETAS的base_rate初始化
19. pnr2目录转换utc时间
20. nsta的pnr2震级相对bgs小于0.15
21. 修改etas的mu的初始化
22. mamba模型的修改
23. 移除所有背景模型的softplus
24. 移除mamba的fc_in_bias和fc_out_bias
25. 解决time_series没有正确mask的问题
26. 更改mu最小值0.2
17. hypernet的context除100


问题
1. f_intensity单调衰减
2. etas似然
3. chuandian rtpp没有对齐
4. etas掩码问题





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
处理# torch.use_deterministic_algorithms(True)结果差异问题：
   增大batch_size
   调低学习率
   更换extractor后效果显著改善，模型结构的问题
scedc不适合做分类
震级预测任务最后一层必须使为类RNN结构，否则last的性能会很差

PE的周期为2pi-2pi *base, 由于有xPos的存在，不会出现表示的混叠
attn相对ssm在长序列tpp没有表现出优势

mhatime在采样阶段只能使用float16
pma效果不如加性注意力
预训练数据集选择对震级预测任务影响较大
chuandian地区震级预测任务的网络深度不应当超过4层

点过程任务经过预训练修改不好

qtmsaltonsea对震级设置非常敏感，震级设置不好会导致标签变化的频率太低

mha_time层自回归是正确的
<!-- 回归任务log优于linear -->



调试
限定g为正值
mag = batch.mag-0.5去除，导致9.6 17664f3 -9.12105d897c512e49f39a294c7bca569240bed5aa11  存在问题 (指标全部有问题)







注意8.12前clf tau_mean 取0.026 ,因为dt没有在Mc过滤后计算




dataset 	Twindow	Tfore	dt	Mf	context_len	rf val rf test	p/n	
ChinaArray	180	60	   10	6.5	1	      0.69	0.6		
ChinaArray	180	30	   10	6.5	1	      0.61	0.53		
ChinaArray	180	90	   10	6.5	1	      0.61	0.53	  0.1	
ChinaArray	180	90	   10	6.5	2	      0.82	0.88   	1	 ~
ChinaArray	180	90	   10	6	   1	      0.76	0.79	   2	~
ChinaArray	180	90	   10	6	   2	      0.93	0.93	   5	
ChinaArray	180	60	   10	6	   1	      0.8	0.69	   1	~
ChinaArray	180	30	   10	6	   1	      0.72	0.5	  0.3	
ChinaArray	180	90	   10	5.5	1	      0.92	0.64	  100	
ChinaArray	180	30	   10	5.5	1	      0.73	0.76	  1	~
ChuanDian	180	10	   10	4  	1	      0.61	0.67	  1          checkpoints/rf_dfae9b13  ./checkpoints/clf_mixer_attnpl_t_20250903-192439
ChuanDian	200	20	   10	4.5	1	      0.64	0.8	  1	
ChuanDian	180	20	   10	4.5	1	      0.61	0.85	  1          checkpoints/rf_847e64da   ./checkpoints/clf_mixer_attnpl_t_20250905-163434
ChuanDian	180	30	   10	4.5	1	      0.65	0.85	  1.5	~      checkpoints/rf_ba2359e6  ./checkpoints/clf_mixer_attnpl_t_20250901-210440
ChuanDian	180	30	   10	5	   1	      0.63	0.8	  0.5	
ChuanDian	180	60	   10	5	   1	      0.88	0.86	  1	~*     checkpoints/rf_9ffe46be  ./checkpoints/clf_mixer_attnpl_t_20250903-103320
ChuanDian	180	60	   10	5.5	1	      0.8	0.69	  0.25	    
ChuanDian	180	60	   10	5.5	2	      0.95	0.67	  0.25	
ChuanDian	180	90	   10	5.5	1	      0.88	0.67	  0.5	~      checkpoints/rf_af684ff4  ./checkpoints/clf_mixer_attnpl_t_20250905-163239
ChuanDian	180	90	   10	5.5	2	      0.96	0.9	  1	
	


600 300
lstm /root/autodl-tmp/chuandian_eq/checkpoints/lstm_20250906-152053
* /root/autodl-tmp/chuandian_eq/checkpoints/reg_mixer_attnpl_t_20250906-194042


消融实验结束
/root/autodl-tmp/chuandian_eq/checkpoints/reg_mixer_attnpl_t_20250907-133839

SCEDC
无log两层 /root/autodl-tmp/chuandian_eq/checkpoints/mixer_tpp_20250907-212136-/root/autodl-tmp/chuandian_eq/checkpoints/mixer_tpp_20250907-212748

clf权重加载实验
checkpoints/clf_mixer_attnpl_t_20250908-142508-checkpoints/clf_mixer_attnpl_t_20250908-143146


AZDX无b：/root/autodl-tmp/chuandian_eq/checkpoints/clf_mixer_attnpl_t_20250909-102301
AZDX有b: /root/autodl-tmp/chuandian_eq/checkpoints/clf_mixer_attnpl_t_20250908-182617
SCEDC无b: /root/autodl-tmp/chuandian_eq/checkpoints/clf_mixer_attnpl_t_20250908-202537
SCEDCb: /root/autodl-tmp/chuandian_eq/checkpoints/clf_mixer_attnpl_t_20250908-183612


SCEDC
tpp: /root/autodl-tmp/chuandian_eq/checkpoints/mixer_tpp_20250907-212619
tpp+b: /root/autodl-tmp/chuandian_eq/checkpoints/mixer_tpp_20250907-212748
tpp+log+b:/root/autodl-tmp/chuandian_eq/checkpoints/mixer_tpp_20250907-190042
tpp+log:  /root/autodl-tmp/chuandian_eq/checkpoints/mixer_tpp_20250907-184838
tpp-txpos: /root/autodl-tmp/chuandian_eq/checkpoints/mixer_tpp_20250909-121137

ChuanDian
tpp+log+b:/root/autodl-tmp/chuandian_eq/checkpoints/mixer_tpp_20250907-193135
tpp+log:  /root/autodl-tmp/chuandian_eq/checkpoints/mixer_tpp_20250907-192418
tpp+b: /root/autodl-tmp/chuandian_eq/checkpoints/mixer_tpp_20250909-114545        用于test
tpp-txpos: checkpoints/mixer_tpp_20250909-123136
tpp /root/autodl-tmp/chuandian_eq/checkpoints/mixer_tpp_20250909-112722

reg     checkpoints/reg_mixer_attnpl_t_20250907-183220
lstm    checkpoints/lstm_20250906-152053
+log    checkpoints/reg_mixer_attnpl_t_20250907-133839
-txpos  checkpoints/reg_mixer_attnpl_t_20250906-205258
-TMAP   checkpoints/reg_mixer_attnpl_t_20250906-211115
-pretrain checkpoints/reg_mixer_attnpl_t_20250906-204638


mixer_tpp 采样成功/root/autodl-tmp/chuandian_eq/checkpoints/mixer_tpp_20250828-102630

rope xpos 消融实验： 
/root/autodl-tmp/chuandian_eq/checkpoints/mixer_tpp_20250912-125450



SCEDC采样绘图，b值预测 ：checkpoints/mixer_tpp_20250912-202038
ChuanDian采样绘图，b值预测 : checkpoints/mixer_tpp_20250912-205124

rotary_embedding_time的scale计算影响采样结果