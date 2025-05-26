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
    cfg_dict['batch_size'] = int(cfg_dict['batch_size'])
    cfg_dict['cuda_id'] = int(cfg_dict['cuda_id'])
    # 如果attn_type字段不存在，改为full
    if 'attn_type' not in cfg_dict:
        cfg_dict['attn_type'] = 'scaled_dot'
    args = SimpleNamespace(**cfg_dict)
    return args


