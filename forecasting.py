# %%
import sys
import argparse
import logging
from pathlib import Path
import torch
import matplotlib.pyplot as plt

from src.utils.utils import set_seed
import src.data.catalog as catalog
import src.catalogs as catalogs
from src.models.builders import ModelBuilder
from src.train.config_setup import load_args_from_checkpoint, load_model_from_checkpoint
from src.utils.visualization import visualize_sequence, visualize_trajectories
from src.utils.logging_utils import setup_logging
import src
import numpy as np

logger = logging.getLogger(__name__)
# %%
def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Run forecast visualization using trained model.")
    parser.add_argument('--dataset', type=str, default='ChuanDian', help='Dataset name')
    parser.add_argument('--t_start', type=int, default=0, help='Forecast start time for past sequence')
    parser.add_argument('--t_forecast', type=int, default=18000, help='Forecast start time')
    parser.add_argument('--duration', type=int, default=30, help='Forecast duration')
    parser.add_argument('--num_samples', type=int, default=1000, help='Total number of samples')
    parser.add_argument('--samples_per_batch', type=int, default=100, help='Samples per batch during sampling')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str,
                        default="./checkpoints/mixer_tpp_20250829-192639",
                        help='Path to checkpoint directory')
    parser.add_argument('--ckpt_select', type=str, choices=['best', 'last', 'epoch'], default='best',
                    help='Which checkpoint to use in test mode (best, last, or epoch)')
    parser.add_argument('--ckpt_epoch', type=int, default=None, help='Epoch number to load when --ckpt_select epoch')
    # "./checkpoints/mixer_tpp_20250826-210727"
    # If args is None, decide based on environment
    if args is None:
        if "ipykernel" in sys.modules:        # Notebook: ignore sys.argv
            return parser.parse_args([])       # use defaults unless you pass a list
        else:                                  # CLI: use real argv
            return parser.parse_args()
    else:
        return parser.parse_args(args)

# %%
args = parse_args()
setup_logging()
set_seed(args.seed)
if args.ckpt_select == 'epoch':
    if args.ckpt_epoch is None:
        raise ValueError('When --ckpt_select is "epoch", --ckpt_epoch must be provided')
    checkpoint_path = Path(args.checkpoint_dir) / f"epoch_{args.ckpt_epoch}_model_1.pth"
else:
    checkpoint_path = Path(args.checkpoint_dir) / f"{args.ckpt_select}_model_1.pth"
check_point = torch.load(checkpoint_path, weights_only=False)
ckpt_args = load_args_from_checkpoint(None, check_point)
logger.info("Loaded checkpoint with args: %s", ckpt_args)
catalog_ds_class = catalog.Catalog.by_name(f"{ckpt_args.dataset}-Standard")
catalog_ds = catalog_ds_class(root_dir=f'data/{ckpt_args.dataset}/raw', catalog_file=None)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_builder = ModelBuilder.by_name(ckpt_args.model.lower())()
model = model_builder(ckpt_args, device)
model, _, _ = load_model_from_checkpoint(model, check_point)
model = torch.compile(model)
model.eval()

test_seq = catalog_ds.test[0]
visualize_sequence(test_seq, show_nll=True)
plt.savefig(f"{args.checkpoint_dir}/sequence_visualization.png", dpi=300, bbox_inches="tight")
plt.close()
# %%
past_seq = test_seq.get_subsequence(args.t_start, args.t_forecast)
observed_seq = test_seq.get_subsequence(args.t_forecast, args.t_forecast + args.duration)
past_batch = src.data.Batch.from_list([past_seq])
#%%
full_seq = test_seq.get_subsequence(args.t_start, args.t_forecast + args.duration)
full_seq.t_nll_start = float(args.t_forecast) 
full_batch = src.data.Batch.from_list([full_seq])
full_batch.to(device)
model.eval()
model.nll_loss(full_batch)['time']
# %%
logger.info("Avg. inter-event time in past_batch: %.4f", torch.mean(past_batch.inter_times).item())

if torch.cuda.is_available():
    model.to('cuda:0')
    past_seq.to('cuda:0')

all_forecasts = []

for _ in range(args.num_samples // args.samples_per_batch):
    forecast = model.sample(
        batch_size=args.samples_per_batch,
        duration=args.duration,
        past_seq=past_seq,
        return_sequences=True
    )
    all_forecasts.extend(forecast)

visualize_trajectories(test_seq, all_forecasts, save_path=Path(args.checkpoint_dir) / "all_forecasts.png")
# %% 
observed_seq.arrival_times
# %%
all_forecasts[0].arrival_times
# %%
t0 = [
    f.arrival_times[0].unsqueeze(-1).to(device)
    for f in all_forecasts
    if len(f.arrival_times) > 0
]
t0_tensor = torch.cat(t0)
mean_t0 = torch.mean(t0_tensor)
# %%
mean_t0
# %%
observed_seq.inter_times
torch.mean(observed_seq.inter_times)
# %%
mean_num_events = np.mean([len(f) for f in all_forecasts])
mean_num_events
# %%
len(observed_seq)
# %%
