# %%
import src
from src.data.preparation import prepare_data_tpp
from config.config_loader import load_args_from_yaml 
args= load_args_from_yaml("config/lstm.yaml")
base_dir = f"data/{args.dataset}"
# %%
prepare_data_tpp(base_dir=base_dir, args=src.args)
# %%
