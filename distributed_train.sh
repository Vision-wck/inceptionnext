#!/bin/bash

# 从脚本的第一个参数 ($1) 获取要使用的进程数，并将其赋值给变量 NUM_PROC。
# 这里的 $1 是通过命令行传递给脚本的第一个参数，通常用来设置训练时使用的 GPU 数量。
NUM_PROC=$1

# 移除脚本参数列表中的第一个参数，即 NUM_PROC。作用是将后续传递给脚本的参数"向左移动"。
# 这意味着原本的 $2 会变成 $1，$3 会变成 $2，依此类推。
# 这样做是为了在后续的命令中将余下的所有参数（如训练参数）传递给 `train.py` 脚本。
shift

# 启动PyTorch的分布式训练。torch.distributed.launch 是 PyTorch 提供的一个模块，用于启动分布式训练任务。
# -m 选项的作用是：让 Python 执行模块，而不仅仅是一个文件
# --nproc_per_node=$NUM_PROC 选项：指定每个节点上启动的进程数。在分布式训练中，通常每个GPU启动一个进程。因此，$NUM_PROC变量应该与可用GPU数量相同
# train.py: 执行训练的脚本
# "$@": 代表所有传递给脚本的参数（除了shift的）
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC train.py "$@"
