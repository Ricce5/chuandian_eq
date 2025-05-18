import pandas as pd
import glob
import numpy as np
import os
import matplotlib.pyplot as plt

def get_split_indices(total_length, train_ratio=0.7, val_ratio=0.15, seed=0, by_time=False):
    indices = np.arange(total_length)
    
    if not by_time:
        np.random.seed(seed)
        indices = np.random.permutation(indices)

    train_len = int(train_ratio * total_length)
    val_len = int(val_ratio * total_length)
    test_len = total_length - train_len - val_len

    train_idx = indices[:train_len]
    val_idx = indices[train_len:train_len + val_len]
    test_idx = indices[train_len + val_len:]

    return train_idx, val_idx, test_idx




def convert_dat_to_csv(root_path):
    """
    递归查找指定目录下所有子目录中的 .dat 文件并将其转换为 .csv 文件。

    参数:
    root_path (str): 根目录路径，程序将从该目录递归查找 .dat 文件。
    """
    # 递归查找所有子目录下的 .dat 文件
    file_list = glob.glob(os.path.join(root_path, "**", "*.dat"), recursive=True)

    for file in file_list:
        # 提取文件名和所在目录
        filename = os.path.basename(file)
        folder = os.path.dirname(file)

        print(f"处理文件: {file}")

        # 读取数据
        df = pd.read_csv(file,
                         delim_whitespace=True,
                         header=None,
                         names=["ID1", "ID2", "Time", "Magnitude", "Depth_m", "Longitude", "Latitude"])

        # 生成输出路径：同目录、同名，只是扩展名改为 .csv
        output_file = os.path.join(folder, filename.replace(".dat", ".csv"))

        # 保存为 .csv 文件
        df.to_csv(output_file, index=False)
        print(f"已保存为: {output_file}")
        
def load_csv_by_keyword(keyword, base_path):
    """
    根据关键字从指定路径加载 CSV 文件，并将其作为 df_<keyword> 的变量存入全局作用域。
    
    参数:
        keyword (str): 匹配文件名中包含的关键字（不带扩展名）
        base_path (str): 要搜索的文件夹路径，默认是 "data/model1/"
    
    返回:
        str: 创建的变量名，或 None 如果未找到文件
    """
    file_list = glob.glob(f"{base_path}/*{keyword}.csv")
    # print(file_list)
    if file_list:
        file_path = file_list[0]
        df = pd.read_csv(file_path)
        print(df.head())
        return df
    else:
        print(f"未找到匹配的 {keyword}.csv 文件")
        return None
    
