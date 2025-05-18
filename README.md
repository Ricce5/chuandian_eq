数据在data文件夹下，代码在src文件夹下，models中的dstpp文件来源于文献Spatio-Temporal Diffusion Point Processes的开源代码。

## 训练

使用以下命令进行模型训练：

```bash
python main.py --model FGCN_f/FGCN/TGCN --mode train --config <config文件路径>
```

如需使用 Optuna 进行超参数调优(在main文件进行超参数范围设置)，将 `--mode` 设置为 `optuna`：

```bash
python main.py --model FGCN_f/FGCN/TGCN --mode optuna --config <config文件路径>

``
## 测试

使用以下命令进行模型测试：

```bash
python main.py --model FGCN_f/FGCN/TGCN --mode test --config <config文件路径> [--checkpoint <checkpoint文件路径>]
```

- 如果未指定 `--checkpoint`，将默认使用最近一次训练的模型。

## 日志

使用 TensorBoard 查看训练日志：

```bash
tensorboard --logdir <checkpoint文件下tensorboard文件夹> --port <端口编号>
```

test.ipynb可用于限定条件查找checkpoint文件