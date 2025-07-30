# %%
import torch
from mamba_ssm.models.config_mamba import MambaConfig
import src
from src.models.mamba.mixer_seq import MambaLMHeadModel

config = MambaConfig(
    d_model=32,
    n_layer=1,
    vocab_size=100,
    d_intermediate=16,
    ssm_cfg={"layer": "Mamba2"},
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    pad_vocab_size_multiple=16,
    tie_embeddings=True,  # 用于绑定 embedding 和 lm_head
)

# 实例化模型（支持 CPU 或 GPU）
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MambaLMHeadModel(config, device=device, dtype=torch.float32).to(device)
model.eval()  # 关闭 dropout，进行推理

input_ids = torch.randint(0, config.vocab_size, (2, 16)).to(device)  # batch=2, seq=16
with torch.no_grad():
    out = model(input_ids=input_ids)
print("Logits (from input_ids):", out.logits.shape)  # (2, 16, vocab_size)

# 先生成一个 embedding 向量
hidden_states = model.backbone.embedding(input_ids)

with torch.no_grad():
    out2 = model.forward_hidden(hidden_states)
print("Logits (from hidden_states):", out2.logits.shape)


diff = (out.logits - out2.logits).abs().max()
print("Max difference between token and hidden input:", diff.item())
# %%
from mamba_ssm.utils.generation import InferenceParams

batch_size = 1
max_seq_len = 128
inference_params = InferenceParams(
    max_seqlen=max_seq_len,
    max_batch_size=batch_size
)

input_ids = torch.tensor([[12, 23, 45, 67]], device=device)  # shape: (1, 4)
generated = []

for i in range(input_ids.shape[1]):
    current_token = input_ids[:, i:i+1].clone().contiguous()  # ✅ clone + contiguous for CUDA conv alignment
    inference_params.seqlen_offset = i

    with torch.no_grad():
        outputs = model(input_ids=current_token, inference_params=inference_params, num_last_tokens=1)
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1)

    generated.append(next_token.item())

print("Generated:", generated)


# %%
