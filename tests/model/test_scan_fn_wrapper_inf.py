import torch
from src.models.mamba.scan_wrapper import SelectiveScanWrapper


def check_full_vs_step_consistency(
    layer: SelectiveScanWrapper,
    B: int = 2,
    L: int = 64,
    D: int = 128,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    atol: float = 1e-5,
    rtol: float = 1e-5,
):
    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    layer = layer.to(device=device)
    layer.eval()

    x = torch.randn(B, L, D, device=device, dtype=dtype)
    delta = torch.rand(B, L, D, device=device, dtype=dtype) * 0.05 + 0.01

    with torch.no_grad():
        y_full, state_full = layer(x, delta, return_state=True)

    ssm_state = layer.allocate_inference_cache(batch_size=B, dtype=torch.float32, device=device)
    ys = []
    with torch.no_grad():
        for t in range(L):
            x_t = x[:, t : t + 1, :]
            dt_t = delta[:, t : t + 1, :]
            y_t, ssm_state = layer.step(x_t, dt_t, ssm_state)
            ys.append(y_t)

    y_step = torch.cat(ys, dim=1)

    diff = (y_full - y_step).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    ok = torch.allclose(y_full, y_step, atol=atol, rtol=rtol)

    # Compare final SSM state from full forward vs incremental step
    state_step = ssm_state.squeeze(2)
    state_match = torch.allclose(state_full, state_step, atol=atol, rtol=rtol)

    print("=== Full vs Step Consistency ===")
    print(f"device={device}, dtype={dtype}, B={B}, L={L}, D={D}")
    print(f"max_abs_diff:  {max_abs:.6e}")
    print(f"mean_abs_diff: {mean_abs:.6e}")
    print(f"allclose(atol={atol}, rtol={rtol}): {ok}")
    print(f"state_allclose(atol={atol}, rtol={rtol}): {state_match}")

    if not ok:
        idx = diff.view(-1).argmax().item()
        b = idx // (L * D)
        rem = idx % (L * D)
        l = rem // D
        d = rem % D
        print(f"worst at (b={b}, l={l}, d={d})")
        print("y_full:", y_full[b, l, d].item())
        print("y_step:", y_step[b, l, d].item())
    if not state_match:
        sd = (state_full - state_step).abs()
        idx = sd.view(-1).argmax().item()
        b = idx // (D * layer.d_state)
        rem = idx % (D * layer.d_state)
        d = rem // layer.d_state
        n = rem % layer.d_state
        print(f"state worst at (b={b}, d={d}, n={n})")
        print("state_full:", state_full[b, d, n].item())
        print("state_step:", state_step[b, d, n].item())

    return ok and state_match, y_full, y_step, state_full, state_step


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = 128
    N = 16
    layer = SelectiveScanWrapper(
        d_model=D,
        d_state=N,
        device=device,
        use_D=True,
        A_max=0,
    )

    check_full_vs_step_consistency(
        layer,
        B=2,
        L=64,
        D=D,
        device=device,
        dtype=torch.float32,
        atol=1e-5,
        rtol=1e-5,
    )
    
