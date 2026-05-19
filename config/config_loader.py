from omegaconf import OmegaConf
from copy import deepcopy
from pathlib import Path

from src.utils.file_utils import create_save_dir, save_args_to_json


def load_args_from_yaml(path='config.yaml'):
    cfg = OmegaConf.load(path)
    if "time_order" in cfg:
        if isinstance(cfg.time_order, list):
            cfg.time_order = tuple(cfg.time_order)
        if set(cfg.time_order) != {"train", "val", "test"} or len(cfg.time_order) != 3:
            raise ValueError(f"time_order must be a permutation of ('train','val','test'), got {cfg.time_order}")
    else:
        cfg.time_order = ("train", "val", "test")

    model_name = str(cfg.model).lower()
    if model_name in {"lstm", "lstm_legacy"}:
        feature_cols = deepcopy(cfg.feature_cols)
        for Mag in cfg.Mag_elaps:
            telaps_key = f"T_elaps{Mag}"
            if telaps_key in feature_cols:
                raise ValueError(f"feature_cols will add feature {telaps_key} from Mag_elaps, duplicate addition is not allowed.")
            feature_cols.append(telaps_key)
        cfg.feature_cols = feature_cols
        print(f"feature_cols: {cfg.feature_cols}")


    if 'B_range' in cfg:
        try:
            cfg.B_range = tuple(map(float, cfg.B_range)) 
            if len(cfg.B_range) != 2 or cfg.B_range[0] >= cfg.B_range[1]:
                raise ValueError(f"B_range should be a tuple of (min, max), got {cfg.B_range}")
        except Exception as e:
            print(f"Error processing B_range: {e}")
            raise

    return cfg


def load_rf_config(project_root: Path, config_relpath: str = "config/rf.yaml"):
    """Load Random Forest config and materialize notebook-friendly runtime args."""
    config_path = Path(project_root) / config_relpath
    args = load_args_from_yaml(str(config_path))

    feature_cols = [
        "Num",
        "Mag_max",
        "Mag_mean",
        "beta",
        "b_lsq",
        "a_lsq",
        "b_std_lsq",
        "std_gr_lsq",
        "b_mlk",
        "a_mlk",
        "b_std_mlk",
        "std_gr_mlk",
        "dM_lsq",
        "dM_mlk",
        "Energy_sqrt",
        "prob_x7_lsq",
        "prob_x7_mlk",
        "zvalue",
    ] + [f"T_elaps{mag}" for mag in args.Mag_elaps]

    args.feature_cols = list(dict.fromkeys(feature_cols))
    args.dMag = 0.1

    args_dict = OmegaConf.to_container(args, resolve=True)
    args.save_dir = create_save_dir(
        str(Path(project_root) / "checkpoints"),
        model_name="rf",
        args_dict=args_dict,
    )

    args_to_save = dict(args_dict)
    args_to_save.pop("save_dir", None)
    for key in [
        "model",
        "batch_size",
        "warmup_ratio",
        "learning_rate",
        "weight_decay",
        "scheduler_type",
        "scheduler_factor",
        "scheduler_patience",
        "scheduler_threshold",
        "scheduler_min_lr",
        "d_model",
        "d_rnn",
        "d_inner",
        "n_layers",
        "n_head",
        "d_k",
        "d_v",
        "t_dropout",
        "attn_type",
        "mlp_out",
        "mlp_dropout",
        "mlp_hdw",
        "epochs",
        "cuda_id",
    ]:
        args_to_save.pop(key, None)

    save_args_to_json(args_to_save, args.save_dir)
    return args
