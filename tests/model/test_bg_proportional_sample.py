# %%
import torch
import src
from src.data.preparation import prepare_data_tpp
from config.config_loader import load_args_from_yaml 
from pathlib import Path
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
# %%
import src.models.bg.proportional as bg_models

# Parameters for testing
B = 5  # Number of parallel samples
d_feature = 2  # Feature dimension for the time series
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a test instance of the model
model = bg_models.ProportionalBGModel(d_feature=d_feature, device=device)

# Generate mock time series data
time_series_times = torch.linspace(0.0, 10.0, 100).unsqueeze(0)  # (1, 100)
# Ensure strictly positive values (uniform in (0,1))
time_series_values = torch.rand(1, 100, d_feature, device=device, dtype=torch.float32)

# Cache the time series data
model.cache_batch(time_series_values, time_series_times)

# Define sample parameters
t0 = torch.tensor([7.0, 3.0, 4.0, 5.0, 6.0], device=device)  # Start times for each sample (B,)
dt = torch.tensor([1.0, 1.5, 2.0, 0.5, 0.1], device=device)  # Duration for each sample (B,)

# Call sample_nhpp_inverse
tau = model.sample_nhpp_inverse(B, t0, dt)

# Print the results
print("Sampled waiting times (tau):", tau)

# %%
# %%
import torch
import src
from src.data.preparation import prepare_data_tpp
from config.config_loader import load_args_from_yaml 
from pathlib import Path
args= load_args_from_yaml("../../config/mixer_tpp.yaml")
# args.dataset = "Geysers"
args.dataset = "PNR"
base_dir = f"../../data/{args.dataset}"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seq, train_loader, val_loader, test_loader, catalog_ds = prepare_data_tpp(base_dir=base_dir, args=args,)
import torch
for batch in train_loader:
    break
# %%
model = bg_models.ProportionalBGModel(d_feature=1, device=device)
model.cache_batch(batch.time_series, batch.time_series_times)
tau = model.sample_nhpp_inverse(20,t0=9.0,dt=9.4)
print("Sampled waiting times (tau):", tau)