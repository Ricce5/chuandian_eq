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
