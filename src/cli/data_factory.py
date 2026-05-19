import importlib

from src.data.preparation import (
    prepare_data,
    prepare_data_lstm,
    prepare_data_lstm_legacy,
    prepare_data_tpp,
)
from src.train.model_routing import get_model_family, get_train_step_module


def get_data_preparation_fn(model_name):
    model_name = model_name.lower()
    if model_name == "lstm":
        return prepare_data_lstm
    if model_name == "lstm_legacy":
        return prepare_data_lstm_legacy

    family = get_model_family(model_name)
    if family in {"classifier", "regressor"}:
        return prepare_data
    return prepare_data_tpp


def get_model_and_data(args, base_path, _device=None):
    model_name = args.model.lower()
    train_step_module = get_train_step_module(model_name)
    train_step = importlib.import_module(f"src.train.{train_step_module}")
    data_func = get_data_preparation_fn(model_name)

    df, train_loader, val_loader, test_loader, _ = data_func(args, base_path)
    return train_step, df, train_loader, val_loader, test_loader
