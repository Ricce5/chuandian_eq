# %%
import os
import json
import yaml
from src.utils.file_utils import find_config_with_conditions

# %%
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler
print(issubclass(ReduceLROnPlateau, LRScheduler))  # 输出 True 表示是子类

# %%
def find_config_with_conditions(conditions, root_dir):
    """
    根据给定的条件筛选 config.yaml 文件，并返回所有符合条件的文件夹绝对路径
    """
    matching_dirs = []

    for subdir, _, files in os.walk(root_dir):
        if 'config.yaml' in files:
            config_path = os.path.join(subdir, 'config.yaml')
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)

                if all(config.get(k) == v for k, v in conditions.items()):
                    matching_dirs.append(os.path.abspath(subdir))
            except Exception as e:
                print(f"[错误] 读取 {config_path} 时失败: {e}")

    return matching_dirs


# %%
root_dir = '/root/autodl-tmp/chuandian_eq/checkpoints'
conditions1 = {
    'Twindow':200,
    'Tfore':20,
    'dt':10,
     'Mc': 4,           # 4,4.5,5.5
     'Mf': 6,
     'context_len':1,
     'split_by_time': True,
     }
conditions2 = {
     'Twindow':30,
    'Tfore':1,
    'dt':10,
     'Mc': 4, 
     'Mf': 5,
     'context_len':3,
     'split_by_time': True,
     }
file_list = find_config_with_conditions(conditions1, root_dir)

# %% [markdown]
# 查找checkpoint

# %%
def get_metrics_json_from_file_list(file_list):
    """
    根据给定的文件列表，检查每个文件夹是否包含 metrics.json 文件，并返回其内容
    
    :param file_list: 包含文件夹路径的列表
    :return: 返回一个包含文件夹路径和对应的 metrics.json 数据的字典列表
    """
    result = [] 

    # 遍历 file_list 中的每个文件夹路径
    for folder_path in file_list:
        metrics_path = os.path.join(folder_path, 'metrics.json')  # 拼接 metrics.json 的路径
        if os.path.isfile(metrics_path):  # 如果 metrics.json 文件存在
            try:
                # 读取并解析 metrics.json 文件
                with open(metrics_path, 'r') as metrics_file:
                    metrics_data = json.load(metrics_file)
                
                # 将文件夹路径和对应的 metrics.json 数据添加到结果列表中
                result.append({
                    'folder_path': os.path.abspath(folder_path),
                    'metrics_data': metrics_data
                })
            except Exception as e:
                print(f"读取 {metrics_path} 文件时发生错误: {e}")
        else:
            print(f"文件夹 {folder_path} 中未找到 metrics.json 文件")

    # 返回符合条件的所有文件夹路径和对应的 metrics.json 数据
    return result


json_files = get_metrics_json_from_file_list(file_list)

# 打印结果
if json_files:
    print("找到的文件夹和对应的 metrics.json 内容:")
    for item in json_files:
        print(f"文件夹路径: {item['folder_path']}")
        print(f"metrics.json 内容: {item['metrics_data']}")
else:
    print("没有找到包含 metrics.json 文件的目录")


# %% [markdown]
# 按日期删除checkpoint

# %%
import os
import shutil
from datetime import datetime

# 设置目标目录
checkpoint_dir = "checkpoints"
# 指定保留起点（目标时间），格式必须和文件夹一致
threshold = "20250518-205722"
threshold_dt = datetime.strptime(threshold, "%Y%m%d-%H%M%S")

# 遍历文件夹
for folder in os.listdir(checkpoint_dir):
    folder_path = os.path.join(checkpoint_dir, folder)
    
    if os.path.isdir(folder_path) and folder.startswith("classifier_"):
        time_str = folder.split("_")[1]
        folder_dt = datetime.strptime(time_str, "%Y%m%d-%H%M%S")

        if folder_dt < threshold_dt:
            print(f"删除：{folder_path}")
            shutil.rmtree(folder_path)  # 删除整个文件夹



