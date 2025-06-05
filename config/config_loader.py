import yaml
import argparse
from types import SimpleNamespace

def load_args_from_yaml(path='config/config.yaml'):
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    # 显式类型转换
    cfg_dict['learning_rate'] = float(cfg_dict['learning_rate'])
    cfg_dict['weight_decay'] = float(cfg_dict['weight_decay'])
    cfg_dict['scheduler_min_lr'] = float(cfg_dict['scheduler_min_lr'])
    if 'scheduler_type' in ['plateau']:
        cfg_dict['scheduler_factor'] = float(cfg_dict['scheduler_factor'])
        cfg_dict['scheduler_patience'] = int(cfg_dict['scheduler_patience'])
        cfg_dict['scheduler_threshold'] = float(cfg_dict['scheduler_threshold'])
    cfg_dict['batch_size'] = int(cfg_dict['batch_size'])
    cfg_dict['cuda_id'] = int(cfg_dict['cuda_id'])

    
    if 'attn_type' not in cfg_dict:
        cfg_dict['attn_type'] = 'scaled_dot'

    if 'time_order' in cfg_dict:
        if isinstance(cfg_dict['time_order'], list):
            cfg_dict['time_order'] = tuple(cfg_dict['time_order'])

        if not isinstance(cfg_dict['time_order'], tuple):
            raise ValueError(f"time_order must be a list or tuple, got {type(cfg_dict['time_order'])}")

        expected = {'train', 'val', 'test'}
        actual = set(cfg_dict['time_order'])
        if actual != expected or len(cfg_dict['time_order']) != 3:
            raise ValueError(f"time_order must be a permutation of ('train', 'val', 'test'), got {cfg_dict['time_order']}")

    else:
        cfg_dict['time_order'] = ('train', 'val', 'test')
    args = SimpleNamespace(**cfg_dict)
    
    if args.model in ["LSTM"]:
        for Mag in args.Mag_elaps:
            telaps_key = f"T_elaps{Mag}"
            if any(telaps_key in feature for feature in args.feature_cols):
                raise ValueError(f"feature_cols会从Mag_elaps添加特征 {telaps_key}，不能重复添加。")
    args.feature_cols.extend([f"T_elaps{Mag}" for Mag in args.Mag_elaps])
    print(f"feature_cols: {args.feature_cols}")
    return args


