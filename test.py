# %%
from omegaconf import OmegaConf

cfg = OmegaConf.create({
    "learning_rate": 0.001,
    "model": {
        "name": "lstm",
        "hidden_dim": 128
    }
})

print(cfg.learning_rate)         # ✅ 输出: 0.001
print(cfg.model.name)            # ✅ 输出: lstm
print(cfg["model"]["hidden_dim"])# ✅ 也_]()

# %%
import torch
checkpoint = torch.load("/root/autodl-tmp/chuandian_eq/checkpoints/mixer_tpp_20250804-195606/last_model_1.pth")
# %%
config = checkpoint['hyperparameters']
# %%
isinstance(config, dict)  # ✅ 检查 config 是否为字典
isinstance(config, OmegaConf)  # ✅ 检查 config 是否为 OmegaConf 对象

# %%
from omegaconf import OmegaConf

# 假设你有一个普通的 dict（container）
container = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "model": {
        "name": "lstm",
        "hidden_dim": 128
    }
}

# 转换为 OmegaConf.DictConfig
cfg = OmegaConf.create(container)

print(type(cfg))          # <class 'omegaconf.dictconfig.DictConfig'>
print(cfg.model.name)     # lstm
dict_cfg =OmegaConf.to_container(cfg, resolve=True)
# %%
print(dict_cfg)
# %%
dict_cfg == container  # ✅ 检查转换后的字典是否与原始字典相同
# %%
checkpoint2 = torch.load("/root/autodl-tmp/chuandian_eq/checkpoints/thp_type_20250622-183014/best_model_1.pth")
# %%
config2 = checkpoint2['hyperparameters']
# %%
