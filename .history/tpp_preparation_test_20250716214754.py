# %%
import src
from src.data.preparation import prepare_data_tpp
from config.config_loader import load_args_from_yaml 
args= load_args_from_yaml("config/thp.yaml")
base_dir = f"data/{args.dataset}"
# %%
seq, train_loader, val_loader, test_loader, catalog_ds = prepare_data_tpp(base_dir=base_dir, args=args,)
# %%
seq.inter_times
# %%
seq.arrival_times
# %%
for batch in train_loader:
    print(batch)
    break
# %%
batch.input_mask
# %%
batch.mag
# %%
batch.arrival_times
# %%
import  src.models.builders as builders
import torch
device = "cuda" if torch.cuda.is_available() else "cpu" 
from src.models.builders import ModelBuilder
model_builder = ModelBuilder.by_name(args.model)()
model = model_builder(args, device)
# %%
print(model)
# %%
model.base_model.input_adapter
# %%
