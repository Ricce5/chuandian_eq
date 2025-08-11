# %%
import src
import torch
from src.data.preparation import prepare_data
from config.config_loader import load_args_from_yaml 
args= load_args_from_yaml("../config/classifier.yaml")
base_dir = f"../data/{args.dataset}"
# %%
seq, train_loader, val_loader, test_loader, catalog_ds = prepare_data(base_dir=base_dir, args=args,)
# %%
for x,y in train_loader:
    break
# %%
x[0,:,:]
torch.max(x[0,:,0])
torch.min(x[0,:,0])
# %%
for x,y in train_loader:
    print(x[0,:,-1])
# %%
for x,y in train_loader:
    print(x[0,:,0])
# %%   
for x,y in train_loader:
    print(x[0,:,0])
    print(x[0,:,-1])
    diff = torch.diff(x[0, :, 0])
    print(diff)
# %%
seq.arrival_times
# %%
for batch in train_loader:
    print(batch)
    break
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
