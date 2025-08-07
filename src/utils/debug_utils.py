import torch
def watch_tensor(name, tensor):
    def hook_fn(grad):
        if torch.isnan(grad).any():
            print(f"❌ NaN in gradient of {name}")
        elif torch.isinf(grad).any():
            print(f"❌ Inf in gradient of {name}")
        else:
            print(f"✅ Gradient of {name} OK: min={grad.min().item():.5f}, max={grad.max().item():.5f}")
    tensor.register_hook(hook_fn)

def check_tensor_anomaly(tensor: torch.Tensor, name="tensor"):
    print(f"[{name}] shape: {tensor.shape}")
    if torch.isnan(tensor).any():
        print(f"  ❗ contains NaN")
    if torch.isinf(tensor).any():
        print(f"  ❗ contains Inf")
    print(f"  min: {tensor.min().item():.4e}")
    print(f"  max: {tensor.max().item():.4e}")
    print(f"  mean: {tensor.mean().item():.4e}")
    print(f"  std: {tensor.std().item():.4e}")
