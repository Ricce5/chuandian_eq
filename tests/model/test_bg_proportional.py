# %%
import torch
import src
from src.data.preparation import prepare_data_tpp
from config.config_loader import load_args_from_yaml 
from pathlib import Path
args= load_args_from_yaml("../../config/mixer_tpp.yaml")
args.dataset = "Geysers"
base_dir = f"../../data/{args.dataset}"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seq, train_loader, val_loader, test_loader, catalog_ds = prepare_data_tpp(base_dir=base_dir, args=args,)
import torch
for batch in train_loader:
    print(batch.arrival_times[:,0])
    print(torch.max(batch.arrival_times,dim=1)[0])

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
bg_model = bg_models.ProportionalBGModel(d_feature=batch.time_series.shape[-1],device=device)
bg_model.to(device)
batch = batch.to(device)
# %%
bg_model.intensity(batch)
# %%
bg_model.intensity_integral(batch).shape
bg_model.to(device)
bg_model.nll_change(batch.to(device), log_h_intensity)
# %%
model.nll_loss(batch.to(device))
# %%
bg_model.cache_batch(batch.time_series, batch.time_series_times)
# %%
time_bg_list, time_bg_tensor = bg_model.sample_nhpp(30000,t0=9.0,t1=9.4,return_times_list=True)
# %%
time_bg_list
# %%
time_bg_tensor 
# %%
tensor1 = torch.tensor([9.8,9.7,9.5]).to(device)
min_tensor = torch.min(tensor1, time_bg_tensor)
# %%
min_tensor
# %%
time_bg_list, time_bg_tensor = bg_model.sample_nhpp(100000,t0=9.0,t1=11,return_times_list=True)
# %%
print(model.bg_model)
# %%
# %%
current_state= current_state.expand(8,-1,-1)
# %%
time_dist = model.get_inter_time_dist(current_state[:,[-1],:])
# %%
model.sample_next_inter_time(inter_time_dist=time_dist)
# %%
model.bg_model = bg_model
# %%
t_last_event = torch.tensor([9.0]*8).to(device)
# %%
model.sample_next_inter_time(inter_time_dist=time_dist, t_last_event=t_last_event)
# %%
model.sample_next_inter_time(inter_time_dist=time_dist, t_last_event=t_last_event,lower_bound=torch.tensor([10]).to(device))

# %%
