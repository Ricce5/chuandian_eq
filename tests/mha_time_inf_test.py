import torch

from src.models.mha.mha_time import MHATime  
from mamba_ssm.modules.mha import MHA
from mamba_ssm.utils.generation import InferenceParams
B, S, D, H = 2, 6000, 64, 4  # batch, seqlen, embed_dim, num_heads
T = torch.arange(S).unsqueeze(0).expand(B, -1).float()  # time indices
from src.utils.utils import set_seed
set_seed(42)


# 创建模型并转换为 float16
model = MHATime(
    embed_dim=D,
    num_heads=H,
    rotary_emb_dim=D // H // 2,
    layer_idx=0,
    causal=True,
    rotary_emb_center_mode="dynamic",
    rotary_emb_scale_base=512,
).to(dtype=torch.float16, device="cuda")


x = torch.randn(B, S, D, dtype=torch.float16, device="cuda")
T = T.to(dtype=torch.float16, device="cuda")  # <== 时间索引也需要同样类型

out_full = model(x, times=T, inference_params=None)

inference_params = InferenceParams(
    max_seqlen=S,
    max_batch_size=B,
    key_value_memory_dict={
        model.layer_idx: model.allocate_inference_cache(batch_size=B, max_seqlen=S)
    },
)
print(out_full.shape)  # 应为 (B, S, D)

outputs = []
for t in range(S):
    token = x[:, t:t+1, :]  # 当前时间步的 token
    dt_step = T[:, t:t+1]    # 当前时间步的时间索引
    out = model(token, times=dt_step, inference_params=inference_params)
    inference_params.seqlen_offset += 1  # 更新序列偏移
    outputs.append(out)

# 拼接逐步推理的输出
output_stepwise = torch.cat(outputs, dim=1)

# 比较一次性前向传播与逐步前向传播的差异
diff = (out_full - output_stepwise).abs().max()  # 最大差异
print(f"full {out_full[0,:,0]}")
print(f"step {output_stepwise[0,:,0]}  ")
print("一次性前向传播与逐步前向传播的最大差异:", diff.item())
assert diff < 1e-4, "一次性前向传播与逐步前向传播结果不一致"
print("✅ 成功：一次性前向传播与逐步前向传播的结果一致！")