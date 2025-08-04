from omegaconf import OmegaConf, DictConfig, ListConfig
import typing

def clean_for_omegaconf(d):
    # 递归清洗，去掉Union类型字段
    if isinstance(d, dict):
        new_d = {}
        for k, v in d.items():
            # 如果v是某个复杂类型，判断是否是Union类型，跳过
            if hasattr(v, '__origin__') and v.__origin__ is typing.Union:
                continue  # 跳过Union类型字段
            if isinstance(v, (dict, list)):
                new_d[k] = clean_for_omegaconf(v)
            else:
                new_d[k] = v
        return new_d
    elif isinstance(d, list):
        return [clean_for_omegaconf(x) for x in d]
    else:
        return d

# 使用示例
restored_args_clean = clean_for_omegaconf(restored_args)
merged_cfg = OmegaConf.merge(OmegaConf.create(base_dict), OmegaConf.create(restored_args_clean))
