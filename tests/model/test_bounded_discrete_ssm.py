import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from src.models.mamba.scan_wrapper import BoundedDiscreteSSM

B_batch, L_len, D_model, D_state = 10, 1000, 4, 16 
B_range = (0.2, 1.0)  
out_range = (0.5,2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ssm_wrapper = BoundedDiscreteSSM(device=device, B_range=B_range,output_range=out_range).to(device)

x_input = torch.randn(B_batch, L_len, D_model, device=device) * 100 
delta_input = torch.abs(torch.randn(B_batch, L_len, D_model, device=device))  

output, ssm_state = ssm_wrapper(x_input, delta_input, return_last_state=True)

print(f"ssm_state shape: {ssm_state.shape}")
print(f"Input shape: {x_input.shape}")
print(f"Output shape: {output.shape}")

output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))



x_input_single = x_input[0,:,0].detach().cpu().numpy()  
output_single = output[0,:,0].detach().cpu().numpy()

time_steps = range(L_len)


plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)   
plt.plot(time_steps, x_input_single, label="Input", color="blue")
plt.title("Input over Time")
plt.xlabel("Time Step")
plt.ylabel("Input Value")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2) 
plt.plot(time_steps, output_single, label="Output", color="red")
plt.title("Output over Time")
plt.xlabel("Time Step")
plt.ylabel("Output Value")
plt.grid(True)
plt.legend()

plt.savefig(os.path.join(output_dir, "input_output_plot.png"))

plt.tight_layout()
plt.show()
