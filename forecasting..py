import sys
import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt

from src.utils.utils import set_seed
import src.data.catalog as catalog
import src.catalogs as catalogs
from src.models.builders import ModelBuilder
from src.train.config_setup import load_args_from_checkpoint, load_model_from_checkpoint
from src.utils.visualization import visualize_sequence, visualize_trajectories
import src

def parse_args():
    parser = argparse.ArgumentParser(description="Run forecast visualization using trained model.")
    parser.add_argument('--dataset', type=str, default='ChuanDian', help='Dataset name')
    parser.add_argument('--t_forecast', type=int, default=18000, help='Forecast start time')
    parser.add_argument('--duration', type=int, default=30, help='Forecast duration')
    parser.add_argument('--num_samples', type=int, default=10000, help='Total number of samples')
    parser.add_argument('--samples_per_batch', type=int, default=1000, help='Samples per batch during sampling')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to checkpoint directory')
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    catalog_ds_class = catalog.Catalog.by_name(f"{args.dataset}-Standard")
    catalog_ds = catalog_ds_class(root_dir=f'data/{args.dataset}/raw', catalog_file=None)

    checkpoint_path = Path(args.checkpoint_dir) / "best_model_1.pth"
    check_point = torch.load(checkpoint_path, weights_only=False)
    ckpt_args = load_args_from_checkpoint(None, check_point)

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

    past_seq = test_seq.get_subsequence(0, args.t_forecast)
    observed_seq = test_seq.get_subsequence(args.t_forecast, args.t_forecast + args.duration)
    past_batch = src.data.Batch.from_list([past_seq])

    print(f"Avg. inter-event time in past_batch: {torch.mean(past_batch.inter_times):.4f}")

    if torch.cuda.is_available():
        model.to('cuda:0')
        past_seq.to('cuda:0')

    past_seq.float()
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

if __name__ == "__main__":
    main()
