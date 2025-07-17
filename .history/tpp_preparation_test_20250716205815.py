# %%
import src
from src.data.preparation import prepare_data_tpp
base_dir = "data/ChuanDian"
# %%
prepare_data_tpp(base_dir=base_dir, args=src.args)