from omegaconf import OmegaConf
from copy import deepcopy

def load_args_from_yaml(path='config.yaml'):
    cfg = OmegaConf.load(path)

    # cfg.learning_rate = float(cfg.learning_rate)
    # cfg.weight_decay = float(cfg.weight_decay)
    # cfg.scheduler_min_lr = float(cfg.scheduler_min_lr)
    # if cfg.scheduler_type == "plateau":
    #     cfg.scheduler_factor = float(cfg.scheduler_factor)
    #     cfg.scheduler_patience = int(cfg.scheduler_patience)
    #     cfg.scheduler_threshold = float(cfg.scheduler_threshold)
    # cfg.batch_size = int(cfg.batch_size)
    # cfg.cuda_id = int(cfg.cuda_id)

    if "time_order" in cfg:
        if isinstance(cfg.time_order, list):
            cfg.time_order = tuple(cfg.time_order)
        if set(cfg.time_order) != {"train", "val", "test"} or len(cfg.time_order) != 3:
            raise ValueError(f"time_order must be a permutation of ('train','val','test'), got {cfg.time_order}")
    else:
        cfg.time_order = ("train", "val", "test")

    if cfg.model == "lstm":
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
