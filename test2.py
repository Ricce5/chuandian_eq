# %%
import torch
buffer_batch2 = torch.load('buffer_batch2.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
buffer_batch2.arrival_times
buffer_batch2.arrival_times[:, 0:1]
#%%
inter_times = torch.randn(100,1).to(device)
mag = torch.randn(100,1).to(device)
buffer_batch2.update_sample_batch(inter_times, mag)
buffer_batch2.get_tmp_batch().arrival_times
# %%
# %%
# %%
