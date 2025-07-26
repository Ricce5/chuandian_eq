import mamba_ssm
from mamba_ssm.modules.mamba import Mamba

print("mamba_ssm version:", mamba_ssm.__version__)
model = Mamba(d_model=64)
print("Mamba layer initialized successfully.")
