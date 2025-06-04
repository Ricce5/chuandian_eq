import os
import re
from datetime import datetime
import pickle
import glob
import yaml
import json
import hashlib

def mkdirs(fn): 
    if not os.path.isdir(fn):
        os.makedirs(fn)
    return fn


def create_save_dir(base_dir, model_name='rf', args_dict=None):
    """
    根据配置参数内容创建唯一目录。
    相同配置 => 生成相同目录
    """
    if args_dict is not None:
        # 将参数字典排序并转为字符串
        config_str = json.dumps(args_dict, sort_keys=True)
        # 使用哈希生成唯一标识
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
    通用的数据缓存加载/保存函数。

    参数:
        base_path (str): 数据根目录路径。
        generate_fn (callable): 当没有缓存时用于生成数据的函数。
        args (tuple): 可选，传递给 generate_fn 的参数。
        filename (str): 可选，保存的文件名。如果未提供，将使用 build_filename 构建。
        sub_dir (str): 可选，子目录路径（相对于 base_path）。
        ext (str): 可选，文件扩展名，默认 "pkl"。
        prefix (str): 可选，生成文件名的前缀。
        **kwargs: 若 filename 为 None，将用于构建文件名。
    
    返回:
        加载或新生成的数据。
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
    
    # 找出所有匹配 model 名的子目录
    all_subdirs = glob.glob(os.path.join(checkpoint_root, "*"))
    matched_subdirs = [
        d for d in all_subdirs
        if os.path.isdir(d) and pattern.search(os.path.basename(d))
    ]

    # 按时间倒序排序（字符串排序即可）
    matched_subdirs.sort(reverse=True)

    # 查找第一个包含模型文件的目录
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
    根据给定的条件筛选 config.yaml 文件，并返回所有符合条件的文件夹绝对路径
    
    :param conditions: 字典，包含多个字段及其对应的值
    :param root_dir: 根目录路径，所有的子目录将被遍历
    :return: 返回符合条件的 config.yaml 文件所在的文件夹绝对路径列表
    """
    matching_dirs = []  # 用于存储符合条件的文件夹路径

    # 遍历 root_dir 目录下的所有子目录和文件
    for subdir, dirs, files in os.walk(root_dir):
        if 'config.yaml' in files:
            config_path = os.path.join(subdir, 'config.yaml')
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    
                    # 检查 config 中是否所有字段都匹配条件
                    if all(config.get(key) == value for key, value in conditions.items()):
                        # 符合条件时，将文件夹的绝对路径加入列表
                        matching_dirs.append(os.path.abspath(subdir))
            except Exception as e:
                print(f"读取 {config_path} 文件时发生错误: {e}")

    # 返回符合条件的所有文件夹路径
    return matching_dirs



def save_args_to_json(args_dict, save_dir, filename="config.json"):
    os.makedirs(save_dir, exist_ok=True)

    # 将 numpy 类型转为原生类型，避免 json 报错
    def convert(o):
        if isinstance(o, (float, int, str, bool)) or o is None:
            return o
        elif hasattr(o, 'tolist'):
            return o.tolist()
        else:
            return str(o)

    with open(os.path.join(save_dir, filename), "w") as f:
        json.dump({k: convert(v) for k, v in args_dict.items()}, f, indent=2)
