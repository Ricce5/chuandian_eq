# %%
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
# 现在再启用 deterministic 模式
torch.use_deterministic_algorithms(True)

import random
import numpy as np

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)
np.random.seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# %%
from src.data.preprocessing import load_and_filter_catalog
import src.data.event_loader as loader 
from config.config_loader import load_args_from_yaml 
import pandas as pd
import numpy as np  
import os
from src.utils.utils import set_seed
set_seed(42)
# %%

args = load_args_from_yaml("config/Classifier.yaml")
base_dir = f"data/{args.dataset}"
df = load_and_filter_catalog(base_dir,Mc=args.Mc)
df_nl = loader.normalize_df(df)
# %%
# samples_list, array_dict = loader.get_list(
#             df, df_nl,
#             Mc=args.Mc, Mf=args.Mf,
#             Twindow=args.Twindow, Tfore=args.Tfore, dt=args.dt,
#             context_len= args.context_len,
#         )
# # %%
# dataset = loader.EventDataset(array_dict, args.Mf)
# # %%
# train_set, val_set, test_set = loader.split_dataset(
#     dataset, by_time=args.split_by_time, train_ratio=0.8, val_ratio=0.1
# )

# train_loader = loader.get_dataloader(train_set, batch_size=args.batch_size, shuffle=True)
# val_loader = loader.get_dataloader(val_set, batch_size=args.batch_size, shuffle=False)
# test_loader = loader.get_dataloader(test_set, batch_size=args.batch_size, shuffle=False)
# %%
import src.data.preparation as preparation
df,train_loader, val_loader, test_loader,_ = preparation.prepare_data(args, base_dir)
# %%
for i, (x, y) in enumerate(test_loader):
    print(x)
    print(y)
    break
# %%
x[:,:,0]
# %%
sequence,train_loader, val_loader, test_loader,_ = preparation.prepare_data_tpp(args, base_dir)
# %%
for batch in train_loader:
    print(batch.inter_times.shape)
    print(batch.arrival_times.shape)
    print(batch.keys())
    break

# %%
import torch
from src.models import Models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import src.train.config_setup as config_setup
args = load_args_from_yaml("config/Classifier.yaml")
args.save_dir = "./checkpoints/classifier_20250520-130712"
checkpoint_path = os.path.join(args.save_dir, "best_model_1.pth")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

args = config_setup.load_args_from_checkpoint(args, checkpoint)
model_class =  getattr(Models,"Classifier")
model, criterion, optimizer, scheduler, args = config_setup.setup_config(
    args, device, model_class,train_loader,
    checkpoint=checkpoint, restore_weights=True
)
# %%
model.eval()
model = model.float().to(device)
x = x.to(dtype=torch.float32).to(device)
# %%
with torch.no_grad():
    print(x)
    output = model(x)
    print(output)
# %%
model.eval()
has_train_modules = False
for name, module in model.named_modules():
    if module.training:
        print(f"[!] {name} still in training mode")
        has_train_modules = True

if not has_train_modules:
    print("[✓] All modules are in eval mode.")

# %%
for name, param in model.named_parameters():
    print(f"{name}: {param}")

# %%
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence

def hook_rnn_(name):
    def hook(module, input, output):
        if isinstance(output, tuple) and isinstance(output[0], PackedSequence):
            packed_out, (h_n, c_n) = output
            out_seq, _ = pad_packed_sequence(packed_out, batch_first=True)

            print(f"[{name}] output sequence shape: {out_seq.data}")
            print(out_seq)

            print(f"[{name}] hidden state h_n shape: {h_n}")
            print(h_n)

            print(f"[{name}] cell state c_n shape: {c_n}")
            print(c_n)
        else:
            print(f"[{name}] Unexpected RNN output: type={type(output)}")
    return hook

model.transformer.rnn_spatial.rnn.register_forward_hook(hook_rnn_("rnn_spatial.rnn"))

with torch.no_grad():
    output = model(x)


# %%
def print_rnn_input(name):
    def hook(module, input, output):
        x = input[0]
        if hasattr(x, 'data'):
            print(f"[{name}] RNN input shape: {x.data.shape}")
            print(f"[{name}] RNN input example: {x.data[0]}")
        else:
            print(f"[{name}] RNN input shape: {x.shape}")
            print(f"[{name}] RNN input example: {x[0]}")
    return hook


# 注册 hook
model.transformer.rnn.rnn.register_forward_hook(print_rnn_input("rnn"))
model.transformer.rnn_temporal.rnn.register_forward_hook(print_rnn_input("rnn_temporal"))
model.transformer.rnn_spatial.rnn.register_forward_hook(print_rnn_input("rnn_spatial"))

# %%

with torch.no_grad():
    output = model(x)
# %%
from flash_attn.modules.mha import FlashSelfAttention

# %%
base_dir = "data/ChuanDian"
args = load_args_from_yaml("config/Regressor.yaml")
df = load_and_filter_catalog(base_dir,Mc=args.Mc)
df_nl = loader.normalize_df(df)
# %%
import src.data.preparation as preparation
df,train_loader, val_loader, test_loader,scalars,dataset = preparation.prepare_data(args, base_dir)
# %%
args= load_args_from_yaml("config/LSTM.yaml")
df,train_loader, val_loader, test_loader,scalars,dataset = preparation.prepare_data_lstm(args, base_dir)
# %%
mag_list = []
for i, (x, y) in enumerate(train_loader):
    mag_list.append(dataset.inverse_normalize_label(y))
for i, (x, y) in enumerate(val_loader):
    mag_list.append(dataset.inverse_normalize_label(y))
for i, (x, y) in enumerate(test_loader):    
    mag_list.append(dataset.inverse_normalize_label(y))
mag = torch.cat(mag_list, dim=0)
if torch.isnan(mag).any():
    print("⚠️ 拼接后的 mag 中存在 NaN 值")
    print(f"NaN 索引位置: {torch.where(torch.isnan(mag))[0]}")
else:
    print("✅ mag 中不含 NaN")
plt.plot(mag.cpu().numpy(), label='Magnitude')

# %%
import matplotlib.pyplot as plt
mag = [dataset.inverse_normalize_label(dataset[i][1]) for i in range(len(dataset))]
plt.plot(mag)

# %%
combined_indices = (
    list(train_loader.dataset.indices) +
    list(val_loader.dataset.indices) +
    list(test_loader.dataset.indices)
)

mag_combined = [dataset.inverse_normalize_label(dataset[i][1]) for i in combined_indices]
plt.plot(mag_combined, label='Combined Magnitude')
# %%
# %%
for i, (x, y) in enumerate(val_loader):
    print(x[:,:,0].shape)
# %%import src.data.preparation as preparation
args.dMag = 0.1
import src.data.preparation as preparation
df,train_loader, val_loader, test_loader,dataset,scalars = preparation.prepare_data_lstm(args, base_dir)

# %%
