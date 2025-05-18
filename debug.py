# %%
from src.data.preprocessing import load_and_filter_catalog
import src.data.classifier_loader as loader 
from config.config_loader import load_args_from_yaml 
import pandas as pd
import numpy as np  
import os

# %%
base_dir = "data/CD2021"
args = load_args_from_yaml("config/Classifier.yaml")
df = load_and_filter_catalog(base_dir,Mc=args.Mc)
df_nl = loader.normalize_df(df)
# %%
samples_list, array_dict = loader.get_list(
            df, df_nl,
            Mc=args.Mc, Mf=args.Mf,
            Twindow=args.Twindow, Tfore=args.Tfore, dt=args.dt,
            context_len= args.context_len,
        )
# %%
dataset = loader.EventDataset(array_dict, args.Mf)
# %%
train_set, val_set, test_set = loader.split_dataset(
    dataset, by_time=args.split_by_time, train_ratio=0.8, val_ratio=0.1
)

train_loader = loader.get_dataloader(train_set, batch_size=args.batch_size, shuffle=True)
val_loader = loader.get_dataloader(val_set, batch_size=args.batch_size, shuffle=False)
test_loader = loader.get_dataloader(test_set, batch_size=args.batch_size, shuffle=False)
# %%
import src.data.preparation as preparation
df,train_loader, val_loader, test_loader = preparation.prepare_data_classifier(args, base_dir)
# %%
