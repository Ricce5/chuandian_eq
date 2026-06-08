

bg_model: mamba
bg_model_cfg:
 d_feature: 1 
 d_state: 64
 d_model: 16
 model_type: mamba  # mamba | mamba2
 scale_init: 300.0
 use_slow_branch: true
 slow_kernel_type: gamma
 slow_kernel_size: 64
 slow_kernel_dt: 1.0
 slow_kernel_normalize: true
 slow_gamma_init_k: 3.0
 slow_gamma_init_beta: 0.3
 fast_mix_init: 0.7
 smooth_kernel_size: 3

