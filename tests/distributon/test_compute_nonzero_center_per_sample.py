import torch

def compute_nonzero_center_per_sample(times: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """
    Compute center from non-zero elements in each sample.
    
    Args:
        times: (batch, seqlen) tensor
        mode: 'mean' | 'midpoint'
        
    Returns:
        (batch,) tensor of center values
    """
    assert mode in ["mean", "midpoint"], f"Unsupported mode: {mode}"

    non_zero_mask = times != 0  # (batch, seqlen)
    print(torch.sum(non_zero_mask, dim=1))
    times_float = times.float()

    if mode == "mean":
        sums = (times_float * non_zero_mask).sum(dim=1)
        counts = non_zero_mask.sum(dim=1).clamp(min=1)  # 避免除0
        center = sums / counts
    elif mode == "midpoint":
        # Set non-zero elements to +inf / -inf and then take min/max
        times_with_inf = times_float.clone()
        times_with_inf[~non_zero_mask] = float('inf')
        min_vals, _ = torch.min(times_with_inf, dim=1)

        times_with_ninf = times_float.clone()
        times_with_ninf[~non_zero_mask] = float('-inf')
        max_vals, _ = torch.max(times_with_ninf, dim=1)

        center = (min_vals + max_vals) / 2

        # If a row is all zeros, min=inf, max=-inf, resulting in nan, which requires additional handling
        all_zero_mask = non_zero_mask.sum(dim=1) == 0
        center[all_zero_mask] = 0 
    return center
def test_compute_nonzero_center_per_sample():
    # Case 1: Normal case with mean mode
    times = torch.tensor([[1.0, 2.0, 0], [3.0, 0, 5], [0, 0, 0]])
    expected_output_mean = torch.tensor([1.5, 4.0, 0.0])  # [(1+2)/2, (3+5)/2, 0]
    output_mean = compute_nonzero_center_per_sample(times, mode="mean")
    assert torch.allclose(output_mean, expected_output_mean), f"Test failed for mean mode. Got {output_mean}"

    # Case 2: Normal case with midpoint mode
    expected_output_midpoint = torch.tensor([1.5, 4.0, 0.0])  # midpoint between min and max non-zero values
    output_midpoint = compute_nonzero_center_per_sample(times, mode="midpoint")
    assert torch.allclose(output_midpoint, expected_output_midpoint), f"Test failed for midpoint mode. Got {output_midpoint}"

    # Case 3: Case with all zeros
    times_all_zeros = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    expected_output_all_zeros = torch.tensor([0.0, 0.0])  # Center is set to 0 for all-zero rows
    output_all_zeros = compute_nonzero_center_per_sample(times_all_zeros, mode="mean")
    assert torch.allclose(output_all_zeros, expected_output_all_zeros), f"Test failed for all-zero input. Got {output_all_zeros}"

    # Case 4: Case with some zeros and some non-zeros
    times_some_zeros = torch.tensor([[0.0, 3.0, 5.0], [4.0, 0.0, 6.0]])
    expected_output_some_zeros = torch.tensor([4.0, 5.0])  # Mean of non-zero values for both rows
    output_some_zeros = compute_nonzero_center_per_sample(times_some_zeros, mode="mean")
    assert torch.allclose(output_some_zeros, expected_output_some_zeros), f"Test failed for some-zero input. Got {output_some_zeros}"

    # Case 5: Case with `midpoint` mode and non-zero values
    expected_output_midpoint_non_zero = torch.tensor([4.0, 5.0])  # Midpoint between min and max non-zero values
    output_midpoint_non_zero = compute_nonzero_center_per_sample(times_some_zeros, mode="midpoint")
    assert torch.allclose(output_midpoint_non_zero, expected_output_midpoint_non_zero), f"Test failed for midpoint with some zeros. Got {output_midpoint_non_zero}"

    # Print success message if all tests pass
    print("All tests passed!")

# Run the test function
test_compute_nonzero_center_per_sample()
