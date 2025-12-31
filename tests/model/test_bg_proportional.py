# %%
import torch
import src
from src.data.preparation import prepare_data_tpp
from config.config_loader import load_args_from_yaml 
from pathlib import Path
args= load_args_from_yaml("../../config/mixer_tpp.yaml")
# args.dataset = "Geysers"
args.dataset = "PNR_1z"
base_dir = f"../../data/{args.dataset}"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seq, train_loader, val_loader, test_loader, catalog_ds = prepare_data_tpp(base_dir=base_dir, args=args,)
import torch
for batch in train_loader:
    print(batch.arrival_times[:,0])
    print(torch.max(batch.arrival_times,dim=1)[0])
#%%
batch_nan_count = int(torch.isnan(batch.time_series).sum().item())
seq_nan_count = int(torch.isnan(seq.time_series).sum().item())
print("batch.time_series NaN count:", batch_nan_count)
print("seq.time_series NaN count:", seq_nan_count)
# %%
checkpoint_dir = Path("../../checkpoints/rtpp_20250819-142340") 
checkpoint_path = checkpoint_dir / "best_model_1.pth"
from src.models.builders import ModelBuilder
from src.train.config_setup import load_args_from_checkpoint,load_model_from_checkpoint

check_point = torch.load(checkpoint_path,weights_only=False)    
args = load_args_from_checkpoint(None, check_point)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_builder = ModelBuilder.by_name(args.model.lower())()
model = model_builder(args, device)
model,_,_ = load_model_from_checkpoint(model, check_point)
# model.set_attn_type("flash")
# model.set_attn_dropout(0)
model = torch.compile(model)
model.eval() 
# %%
model(batch.to(device))
# %%
current_state = model.get_context(batch)
current_state = current_state.expand(1, -1, -1) 
time_dist = model.get_inter_time_dist(current_state)
# %%
log_h_intensity = time_dist.log_hazard(batch.inter_times[:,0].to(device))
# %%
import src.models.bg.proportional as bg_models
bg_model = bg_models.ProportionalBGModel(d_feature=batch.time_series.shape[-1],scale_init=200,device=device)
bg_model.to(device)
batch = batch.to(device)
# %%
bg_model.intensity(batch)
# %%
torch.isnan(bg_model.intensity(batch)).any()
# %%
ts = batch.arrival_times
ts_times = batch.time_series_times
t_query = batch.arrival_times
print("t.shape, x.shape, t_query.shape:", ts_times.shape, batch.time_series.shape, t_query.shape)
print("t0:", ts_times[:, 0:2])               # 查看前两个时间点
dt = (ts_times[:,1] - ts_times[:,0])
print("dt:", dt)
print("any dt==0:", (dt == 0).any().item())
print("any NaN in x:", torch.isnan(batch.time_series).any().item())
print("t_query min/max:", t_query.min(), t_query.max())
print("t range min/max:", ts_times[:,0].min(), ts_times[:,-1].max())
from src.utils.interp import interp_uniform_time_series
result    = interp_uniform_time_series(
            t=ts_times,      # (B, T)
            x=batch.time_series,            # (B, T, F)
            t_query=t_query,
            )
print(result)
# %%
bg_model.intensity_integral(batch).shape
bg_model.to(device)
bg_model.nll_change(batch.to(device), log_h_intensity)
# %%
model.nll_loss(batch.to(device))
# %%
bg_model.cache_batch(batch.time_series, batch.time_series_times)
# %%
t0 = 9.0
times_list= bg_model.sample_nhpp_inverse(3000,t0=9.0,dt=20,sample_sequence=True)
tau2 =  bg_model.sample_nhpp_inverse(3000,t0=9.0,dt=20,sample_sequence=False)
# %%
# %%
# 安全地取每个序列的第一个时间点，缺失则填 NaN
tau1_list = []
for times in times_list:
        tau1_list.append(torch.tensor(times[0]-t0).to(device))
tau1 = torch.stack(tau1_list)
torch.mean(tau2)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
tau1_np = tau1.cpu().numpy()
tau2_np = tau2.cpu().numpy()

# Create a figure for plotting
plt.figure(figsize=(12, 6))

# Plot the histogram and KDE for tau1
sns.histplot(tau1_np, kde=True, color='blue', label='tau1', stat="density", linewidth=2)

# Plot the histogram and KDE for tau2
sns.histplot(tau2_np, kde=True, color='red', label='tau2', stat="density", linewidth=2)

# Adding title and labels
plt.title('Distribution of tau1 and tau2', fontsize=14)
plt.xlabel('Value', fontsize=12)
plt.ylabel('Density', fontsize=12)

# Add a legend
plt.legend()

# Show the plot
plt.show()
# %%
from scipy.stats import ks_2samp
alpha = 0.01

# 进行 Kolmogorov-Smirnov 检验
stat, p_value = ks_2samp(tau1.cpu().numpy(), tau2.cpu().numpy())

print(f"KS-statistic: {stat}")
print(f"p-value: {p_value}")

# 根据 p-value 判断显著性
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference.")
else:
    print("Fail to reject the null hypothesis: No significant difference.")

# %%
times_list[0]
# %%
times_list[1]
# %%
