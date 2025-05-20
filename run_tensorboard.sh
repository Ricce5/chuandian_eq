#!/bin/bash

LOGDIR=$1  # 第一个参数是 logdir
PORT_START=${2:-7007}  # 第二个参数是起始端口，默认 7007
PORT_END=${3:-7099}    # 第三个参数是最大端口，默认 7099

if [ -z "$LOGDIR" ]; then
  echo "用法: ./run_tensorboard.sh <logdir> [start_port] [end_port]"
  exit 1
fi

# 查找空闲端口
for ((port=$PORT_START; port<=$PORT_END; port++)); do
  if ! lsof -iTCP:$port -sTCP:LISTEN >/dev/null 2>&1; then
    echo "找到空闲端口: $port"
    echo "启动 TensorBoard..."
    tensorboard --logdir "$LOGDIR" --port $port
    exit 0
  fi
done

echo "未找到空闲端口（$PORT_START 到 $PORT_END）之间可用端口。"
exit 1
