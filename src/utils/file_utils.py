import os
import re
from datetime import datetime
import pandas as pd
import pickle
import glob
import yaml
import json
import hashlib
from  omegaconf import OmegaConf

def mkdirs(fn): 
    if not os.path.isdir(fn):
        os.makedirs(fn)
    return fn


def create_save_dir(base_dir, model_name='rf', args_dict=None):
    """
    Create a unique directory based on configuration parameters.
    Same configuration => Same directory
    """
    if args_dict is not None:
        config_str = json.dumps(args_dict, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()[:8]
        dir_name = f"{model_name}_{config_hash}".lower()
    else:
        from datetime import datetime
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        dir_name = f"{model_name}_{time_str}".lower()

    save_dir = os.path.join(base_dir, dir_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def build_filename(prefix="processed", ext="pkl", **kwargs):
    def format_value(val):
        if isinstance(val, list):
            return '-'.join(map(str, val))
        if val is None:
            return "none"
        if isinstance(val, bool):
            return str(val).lower()
        return str(val)

    parts = [f"{key}_{format_value(kwargs[key])}" for key in sorted(kwargs)]
    filename = f"{prefix}_{'_'.join(parts)}.{ext}" if parts else f"{prefix}.{ext}"
    return filename

def save_or_load_data(base_path, generate_fn, args=None, filename=None, sub_dir="processed", ext="pkl", prefix="data", **kwargs):
    """
    General-purpose data caching load/save function.

    Parameters:
        base_path (str): Root directory path for data.
        generate_fn (callable): Function to generate data when no cache exists.
        args (tuple): Optional, arguments to pass to generate_fn.
        filename (str): Optional, name of the saved file. If not provided, build_filename will be used to construct it.
        sub_dir (str): Optional, subdirectory path (relative to base_path).
        ext (str): Optional, file extension, default is "pkl".
        prefix (str): Optional, prefix for the generated filename.
        **kwargs: If filename is None, these will be used to construct the filename.
    
    Returns:
        Loaded or newly generated data.
    """
    if filename is None:
        filename = build_filename(prefix=prefix, ext=ext, **kwargs)
    elif not filename.endswith(f".{ext}"):
        filename += f".{ext}"

    save_dir = os.path.join(base_path, sub_dir)
    os.makedirs(save_dir, exist_ok=True)
    full_path = os.path.join(save_dir, filename)

    if os.path.exists(full_path):
        print(f"[✔] Loading cached data from: {full_path}")
        with open(full_path, "rb") as f:
            return pickle.load(f)
    else:
        print(f"[✱] Processing and saving new data to: {full_path}")
        data = generate_fn(*(args or ()))
        with open(full_path, "wb") as f:
            pickle.dump(data, f)
        return data



def find_latest_model_path(model_name, checkpoint_root="checkpoints"):
    pattern = re.compile(rf"{model_name.lower()}_\d{{8}}-\d{{6}}$")
    
    # find all subdirectories matching the pattern
    all_subdirs = glob.glob(os.path.join(checkpoint_root, "*"))
    matched_subdirs = [
        d for d in all_subdirs
        if os.path.isdir(d) and pattern.search(os.path.basename(d))
    ]

    # sort by modification time, newest first
    matched_subdirs.sort(reverse=True)

    # check for the existence of last_model_1.pth in each matched subdirectory
    for subdir in matched_subdirs:
        candidate = os.path.join(subdir, "last_model_1.pth")
        if os.path.isfile(candidate):
            print(f"Found folder: {subdir}")
            return os.path.abspath(subdir)

    raise FileNotFoundError(f"No checkpoint found for model: {model_name}")


def remove_empty_dirs(root_path):
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        if not dirnames and not filenames:
            os.rmdir(dirpath)

def create_unique_dir(base_path): 
    remove_empty_dirs(base_path)
    
    sub_folder_name = re.sub(r'[^0-9]', '', str(datetime.now()))
    unique_path = os.path.join(base_path, sub_folder_name)
    mkdirs(unique_path)
    return unique_path


def find_config_with_conditions(conditions, root_dir):
    """
    Filter config.yaml files based on given conditions and return the absolute paths of all matching directories.
    
    :param conditions: Dictionary containing multiple fields and their corresponding values.
    :param root_dir: Root directory path, all subdirectories will be traversed.
    :return: A list of absolute paths to directories containing config.yaml files that meet the conditions.
    """
    matching_dirs = []  

    # Traverse all subdirectories and files under root_dir
    for subdir, dirs, files in os.walk(root_dir):
        if 'config.yaml' in files:
            config_path = os.path.join(subdir, 'config.yaml')
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    
                    if all(config.get(key) == value for key, value in conditions.items()):
                        matching_dirs.append(os.path.abspath(subdir))
            except Exception as e:
                print(f"Error reading {config_path}: {e}")
    return matching_dirs

def load_csv_by_keyword(keyword, base_path):
    """
    Load a CSV file based on a keyword from the specified path and return it as a DataFrame.
    
    Parameters:
        keyword (str): The keyword to match in the file name (without extension).
        base_path (str): The folder path to search in.
    
    Returns:
        pd.DataFrame: The loaded DataFrame, or None if no matching file is found.
    """
    file_list = glob.glob(f"{base_path}/*{keyword}.csv")
    if file_list:
        file_path = file_list[0]
        df = pd.read_csv(file_path)
        print(df.head())
        return df
    else:
        print(f"No matching {keyword}.csv file found")
        return None


def save_args_to_json(args_dict, save_dir, filename="config.json"):
    os.makedirs(save_dir, exist_ok=True)

    # Convert numpy types to native types to avoid json errors
    def convert(o):
        if isinstance(o, (float, int, str, bool)) or o is None:
            return o
        elif hasattr(o, 'tolist'):
            return o.tolist()
        else:
            return str(o)

    with open(os.path.join(save_dir, filename), "w") as f:
        json.dump({k: convert(v) for k, v in args_dict.items()}, f, indent=2)

def build_catalog_root_dir(base_root_dir, catalog_cfg):
        try:
            cfg_container = OmegaConf.to_container(OmegaConf.create(catalog_cfg), resolve=True)
        except Exception:
            cfg_container = catalog_cfg

        cfg_str = json.dumps(cfg_container, sort_keys=True, separators=(",", ":"))
        cfg_hash = hashlib.md5(cfg_str.encode("utf-8")).hexdigest()[:8]

        cfg_dir = f"{cfg_hash}"
        root_dir = os.path.join(base_root_dir, cfg_dir)
        os.makedirs(root_dir, exist_ok=True)

        cfg_save_path = os.path.join(root_dir, "catalog_cfg.json")
        with open(cfg_save_path, "w") as f:
            json.dump(cfg_container, f, indent=2, sort_keys=True)
        return root_dir, cfg_container