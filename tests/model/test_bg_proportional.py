# %%
import src
from src.data.preparation import prepare_data_tpp
from config.config_loader import load_args_from_yaml 
args= load_args_from_yaml("/home/yzzhang/pjt/chuandian_eq/config/mixer_tpp.yaml")
args.dataset = "Geysers"
base_dir = f"../../data/{args.dataset}"

seq, train_loader, val_loader, test_loader, catalog_ds = prepare_data_tpp(base_dir=base_dir, args=args,)
import torch
for batch in train_loader:
    print(batch.arrival_times[:,0])
    print(torch.max(batch.arrival_times,dim=1)[0])
# %%
from  src.models.bg.bg_proportional import ProportionalBGModel
model = ProportionalBGModel(d_feature=batch.time_series.shape[-1])
# %%
model.intensity(batch)
# %%
model.intensity_integral(batch).shape
# %%
