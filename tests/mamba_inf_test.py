import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.utils.generation import InferenceParams

device = "cuda"
dtype = torch.float16
seqlen = 16   # full sequence length
gen_len = 8   # how many tokens to generate
batch_size = 1


config = MambaConfig(
    d_model=512,
    n_layer=2,
    vocab_size=1000,
    ssm_cfg={"layer": "Mamba2"},
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
)
model = MambaLMHeadModel(config, device=device, dtype=dtype)


prompt = torch.randint(0, 1000, (batch_size, seqlen - gen_len), device=device)

inference_params = InferenceParams(
    max_seqlen=seqlen,
    max_batch_size=batch_size,
    key_value_memory_dict=model.allocate_inference_cache(batch_size, seqlen),
)

logits = model(prompt, inference_params=inference_params).logits
print("Initial logits (prompt):", logits.shape)

generated = []
input_token = prompt[:, -1:]  # 从 prompt 最后一个 token 开始
for step in range(gen_len):
    inference_params.seqlen_offset += 1
    logits = model(input_token, inference_params=inference_params, num_last_tokens=1).logits
    next_token = logits.argmax(dim=-1)  # greedy decode
    generated.append(next_token)
    input_token = next_token  # feed it back



generated = torch.cat(generated, dim=1)
output_sequence = torch.cat([prompt, generated], dim=1)

print("Generated sequence shape:", output_sequence.shape)
print("Output sequence:", output_sequence)
