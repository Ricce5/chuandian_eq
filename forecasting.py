# %%
import sys
import argparse
import logging
from pathlib import Path
import torch
import matplotlib.pyplot as plt

from src.utils.utils import set_seed
import src.data.catalog as catalog
from src.models.builders import ModelBuilder
from src.train.config_setup import load_args_from_checkpoint, load_model_from_checkpoint
from src.utils.visualization import visualize_sequence, visualize_trajectories
from src.utils.logging_utils import setup_logging
import src
import numpy as np
from src.catalogs.pathing import build_tpp_catalog_init_kwargs

logger = logging.getLogger(__name__)
# %%
def parse_args(cli_args=None):
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
    # If cli_args is None, decide based on environment.
    if cli_args is None:
        if "ipykernel" in sys.modules:        # Notebook: ignore sys.argv
            return parser.parse_args([])       # use defaults unless you pass a list
        else:                                  # CLI: use real argv
            return parser.parse_args()
    else:
        return parser.parse_args(cli_args)

# %%
args = parse_args()
setup_logging()
set_seed(args.seed)
if args.ckpt_select == 'epoch':
    if args.ckpt_epoch is None:
        raise ValueError('When --ckpt_select is "epoch", --ckpt_epoch must be provided')
    checkpoint_file = Path(args.checkpoint_dir) / f"epoch_{args.ckpt_epoch}_model_1.pth"
else:
    checkpoint_file = Path(args.checkpoint_dir) / f"{args.ckpt_select}_model_1.pth"
checkpoint_data = torch.load(checkpoint_file, weights_only=False)
checkpoint_args = load_args_from_checkpoint(None, checkpoint_data)
logger.info("Loaded checkpoint with args: %s", checkpoint_args)


def build_catalog_from_checkpoint(dataset_name: str):
    cls = catalog.Catalog.by_name(f"{dataset_name}-Standard")
    base_dir = Path("data") / dataset_name
    init_kwargs = build_tpp_catalog_init_kwargs(
        catalog_ds_class=cls,
        dataset_name=dataset_name,
        base_dir=base_dir,
        catalog_cfg=None,
    )

    return cls(**init_kwargs)


dataset_catalog = build_catalog_from_checkpoint(checkpoint_args.dataset)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_builder = ModelBuilder.by_name(checkpoint_args.model.lower())()
model = model_builder(checkpoint_args, device)
model, _, _ = load_model_from_checkpoint(model, checkpoint_data)
model = torch.compile(model)
model.eval()

test_sequence = dataset_catalog.test[0]
visualize_sequence(test_sequence, show_nll=True)
plt.savefig(f"{args.checkpoint_dir}/sequence_visualization.png", dpi=300, bbox_inches="tight")
plt.close()
# %%
past_sequence = test_sequence.get_subsequence(args.t_start, args.t_forecast)
observed_sequence = test_sequence.get_subsequence(args.t_forecast, args.t_forecast + args.duration)
past_batch = src.data.Batch.from_list([past_sequence])
#%%
full_sequence = test_sequence.get_subsequence(args.t_start, args.t_forecast + args.duration)
full_sequence.t_nll_start = float(args.t_forecast) 
full_batch = src.data.Batch.from_list([full_sequence])
full_batch.to(device)
model.eval()
model.nll_loss(full_batch)['time']
# %%
logger.info("Avg. inter-event time in past_batch: %.4f", torch.mean(past_batch.inter_times).item())

if torch.cuda.is_available():
    model.to(device)
    past_sequence.to(device)

forecast_sequences = []

num_sampling_batches = args.num_samples // args.samples_per_batch
for _ in range(num_sampling_batches):
    sampled_sequences = model.sample(
        batch_size=args.samples_per_batch,
        duration=args.duration,
        past_seq=past_sequence,
        return_sequences=True
    )
    forecast_sequences.extend(sampled_sequences)

visualize_trajectories(test_sequence, forecast_sequences, save_path=Path(args.checkpoint_dir) / "all_forecasts.png")
# %% 
observed_sequence.arrival_times
# %%
forecast_sequences[0].arrival_times
# %%
first_arrival_times = [
    sequence.arrival_times[0].unsqueeze(-1).to(device)
    for sequence in forecast_sequences
    if len(sequence.arrival_times) > 0
]
first_arrival_tensor = torch.cat(first_arrival_times)
mean_first_arrival = torch.mean(first_arrival_tensor)
# %%
mean_first_arrival
# %%
observed_sequence.inter_times
torch.mean(observed_sequence.inter_times)
# %%
mean_num_events = np.mean([len(sequence) for sequence in forecast_sequences])
mean_num_events
# %%
len(observed_sequence)
# %%
