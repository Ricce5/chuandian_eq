import torch
from mamba_ssm.ops.triton.selective_state_update import selective_state_update ,selective_state_update_ref
# 设定参数
batch = 2
nheads = 2
dim = 4
dstate = 8
ngroups = 1  # 保证 nheads % ngroups == 0

# 构造最简张量，全部为 float32 类型
state = torch.randn(batch, nheads, dim, dstate, device='cuda')
x = torch.randn(batch, nheads, dim, device='cuda')
dt = torch.rand(batch, nheads, dim, device='cuda')  # 保证非负
dt_bias = torch.zeros(nheads, dim, device='cuda')  # 保证非负
A = torch.randn(nheads, dim, dstate, device='cuda')
B = torch.randn(batch, ngroups, dstate, device='cuda')
C = torch.randn(batch, ngroups, dstate, device='cuda')

# 可选项设为 None
D = torch.randn(nheads, dim, device='cuda') if ngroups > 0 else None
z = None

# 不使用 dt_softplus
dt_softplus = False

# 模拟函数调用
try:
    out = selective_state_update(
        state=state,
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
    )
    print("✅ 测试成功，无报错。输出 shape:", out.shape)
    out_ref = selective_state_update_ref(
        state=state,
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
    )
    diff = (out - out_ref).abs().max()
    print("最大差异:", diff.item())
    assert diff < 1e-5, "输出与参考实现不一致！"
except Exception as e:
    print("❌ 报错:", e)
