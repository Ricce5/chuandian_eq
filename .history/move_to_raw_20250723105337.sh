#!/bin/bash

# 设置目录路径
BASE_DIR="/home/yzzhang/pjt/chuandian_eq/data/QTMSaltonSea"
RAW_DIR="$BASE_DIR/raw"

# 1. 创建 raw 子目录（如果不存在）
mkdir -p "$RAW_DIR"

# 2. 移动除了 raw 目录自身以外的所有内容
shopt -s extglob  # 开启高级通配模式
mv "$BASE_DIR"/!(raw) "$RAW_DIR"

echo "✅ 所有文件已移动到 $RAW_DIR"
