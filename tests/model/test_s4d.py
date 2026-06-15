import torch

from src.models.mamba.s4d import S4D


def test_s4d_fft_matches_recurrent_steps():
    torch.manual_seed(0)
    model = S4D(
        d_model=4,
        d_state=8,
        dropout=0.0,
        activation="identity",
        use_output_linear=False,
    )
    x = torch.randn(2, 17, 4)

    y_fft, final_state = model(x, return_state=True)
    step_state = model.allocate_inference_cache(batch_size=x.size(0), dtype=x.dtype, device=x.device)
    y_steps = []
    for t in range(x.size(1)):
        y_t, step_state = model.step(x[:, t : t + 1, :], step_state)
        y_steps.append(y_t)
    y_step = torch.cat(y_steps, dim=1)

    assert torch.allclose(y_fft, y_step, atol=1e-5, rtol=1e-5)
    assert torch.allclose(final_state, step_state, atol=1e-5, rtol=1e-5)


def test_s4d_default_shape_and_kernel_return():
    torch.manual_seed(0)
    model = S4D(d_model=3, d_state=8, dropout=0.0, input_norm=True, input_linear=True)
    x = torch.randn(2, 13, 3)

    y, kernel = model(x, return_kernel=True)

    assert y.shape == x.shape
    assert kernel.shape == (3, 13)


def test_s4d_stateful_forward_matches_steps():
    torch.manual_seed(0)
    model = S4D(
        d_model=3,
        d_state=8,
        dropout=0.0,
        activation="identity",
        use_output_linear=False,
    )
    x = torch.randn(2, 9, 3)
    init_state = torch.randn(2, 3, 4, dtype=torch.complex64)

    y_recurrent, final_state = model(x, ssm_state=init_state.clone(), return_state=True)
    step_state = init_state.clone()
    y_steps = []
    for t in range(x.size(1)):
        y_t, step_state = model.step(x[:, t : t + 1, :], step_state)
        y_steps.append(y_t)
    y_step = torch.cat(y_steps, dim=1)

    assert torch.allclose(y_recurrent, y_step, atol=1e-5, rtol=1e-5)
    assert torch.allclose(final_state, step_state, atol=1e-5, rtol=1e-5)

