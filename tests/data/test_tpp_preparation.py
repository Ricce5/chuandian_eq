# %%
import src
from src.data.preparation import prepare_data_tpp
from config.config_loader import load_args_from_yaml 
args= load_args_from_yaml("/home/yzzhang/pjt/chuandian_eq/config/mixer_tpp.yaml")
args.dataset = "Geysers"
base_dir = f"../../data/{args.dataset}"
# %%
seq, train_loader, val_loader, test_loader, catalog_ds = prepare_data_tpp(base_dir=base_dir, args=args,)
# %%
seq.inter_times
# %%
seq.arrival_times
# %%
import torch
for batch in train_loader:
    print(batch.arrival_times[:,0])
    print(torch.max(batch.arrival_times,dim=1)[0])
# %%
import math
from src.distributions.gamma import Gamma
dist = Gamma(batch.a_t, batch.s_t * math.log(10.0))
dist.log_prob(batch.b_mean)
# %%
batch.input_mask.shape
# %%
batch.mag.shape
# %%
batch.arrival_times.shape
# %%
batch.inter_times.shape
# %%
batch.nll_event_mask.shape
# %%
import  src.models.builders as builders
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from src.models.builders import ModelBuilder
model_builder = ModelBuilder.by_name(args.model)()
model = model_builder(args, device)
# %%
print(model)
# %%
model.base_model.input_adapter
# %%
model(batch.to(device))[0].shape
# %%
import numpy as np
inter_times = np.load('inter_times.npz')['inter_times']
arrival_times = np.load('arrival_times.npz')['arrival_times']   
# %%
neg_indices = np.argwhere(inter_times < 0)  # coordinates of elements < 0
neg_indices
# %%
len(inter_times)
# %%
inter_times
# %%
batch.time_series.shape
# %%
batch.arrival_times
# %%
t = batch.time_series_times # (B, T)
x = batch.time_series       # (B, T, F)
t_query = batch.arrival_times # (B, Nq)



# %%
import torch
from src.utils.interp import  interp_uniform_time_series, integrate_uniform_time_series
# %%
x_query = interp_uniform_time_series(
    t=batch.time_series_times,      # (B, T)
    x=batch.time_series,            # (B, T, F)
    t_query=batch.arrival_times,    # (B, Nq)
)

# %%
integral = integrate_uniform_time_series(
    t=batch.time_series_times,      # (B, T)
    x=batch.time_series,            # (B, T, F)
    t_start= batch.t_nll_start,  # (B,)
    t_end=batch.arrival_times[:,-1],      # (B,)
)
# %%
