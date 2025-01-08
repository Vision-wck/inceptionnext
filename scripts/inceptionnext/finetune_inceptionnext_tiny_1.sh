#!/bin/bash

# 定义数据路径，指向包含图像数据的目录
DATA_PATH=/home/joey.wang/pz_project/xingsen/smart_precheck_system/PCBlayer_name_recognize/data/PCB_image/20241226/class_split
# 定义代码路径，指向inceptionnext项目代码所在目录
CODE_PATH=/home/joey.wang/Vision/inceptionnext
# 定义预训练模型的检查点路径，指向一个预训练模型文件，用于初始化模型的权重
INIT_CKPT=/home/joey.wang/Vision/inceptionnext/pretrained/inceptionnext_tiny.pth

CUDA_VISIBLE_DEVICES=4,5

# 定义全批次大小，表示训练中使用的总批次大小，不考虑GPU数量和梯度累积
ALL_BATCH_SIZE=512
# 定义使用的GPU数量，这是为了进行分布式训练而设置的GPU数量
NUM_GPU=2
# 定义梯度累积步数。如果内存有限，可以通过梯度累积来模拟更大的批次
GRAD_ACCUM_STEPS=2 # Adjust according to your GPU numbers and memory size.
# 计算每个GPU上的批次大小，
# 使用总批次大小除以GPU数量和梯度累积步数，以确保每个GPU上的批次适配并节省内存
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS


MODEL=inceptionnext_tiny
# 设置DropPath的概率。DropPath是一种正则化技术，用于随机丢弃神经网络中的路径
DROP_PATH=0.7
# 设置Drop的概率。这是另一种正则化方法，用于丢弃神经网络中一些连接
DROP=0.5

OUTPUT=./output
#EXPERIMENT=exp0

EPOCHS=200

echo "Begining"


# 进入代码目录，并执行distributed_train.sh脚本来启动分布式训练
cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
  --model $MODEL \
  --initial-checkpoint $INIT_CKPT \
  --img-size 224 \
  --epochs $EPOCHS \
  --opt adamw \
  --lr 5e-5 \
  --sched None \
  -b $BATCH_SIZE \
  --grad-accum-steps $GRAD_ACCUM_STEPS \
  --mixup 0 \
  --cutmix 0 \
  --drop-path $DROP_PATH \
  --drop $DROP \
  --output $OUTPUT \
  --workers 16 \

#cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
#  --model $MODEL \
#  --initial-checkpoint $INIT_CKPT \
#  --img-size 224 \
#  --epochs $EPOCHS \
#  --opt adamw \
#  --lr 5e-5 \
#  --sched None \
#  -b $BATCH_SIZE \
#  --grad-accum-steps $GRAD_ACCUM_STEPS \
#  --mixup 0 \
#  --cutmix 0 \
#  --model-ema \
#  --model-ema-decay 0.9999 \
#  --drop-path $DROP_PATH \
#  --drop $DROP \
#  --output $OUTPUT \
#  --workers 8 \


# 参数注释
#cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
#  # 传递模型类型和参数
#  --model $MODEL \      # 指定使用的模型架构
#  --img-size 224 \      # 指定输入图像的尺寸
#  --epochs 30 \         # 设置训练的总 epochs 数
#  --opt adamw \         # 使用 AdamW 优化器（适用于更稳定的训练）
#  --lr 5e-5 \           # 设置学习率为 5e-5
#  --sched None \        # 不使用学习率调度器（可以根据需求选择）
#
#  # 设置训练中的批次大小和梯度累积步数
#  -b $BATCH_SIZE \      # 设置每个梯度累积步骤中的批次大小
#  --grad-accum-steps $GRAD_ACCUM_STEPS \    # 设置梯度累积步数
#
#  # 提供预训练模型权重的路径（用于初始化模型）
#  --initial-checkpoint $INIT_CKPT \
#
#  # 关闭 MixUp 和 CutMix（数据增强方法）
#  --mixup 0 \     # 禁用 MixUp 数据增强
#  --cutmix 0 \    # 禁用 CutMix 数据增强
#
#  # 启用模型的 EMA（Exponential Moving Average）更新，并设置EMA衰减率
#  --model-ema \
#  --model-ema-decay 0.9999 \
#
#  # 配置DropPath和Drop的正则化设置
#  --drop-path $DROP_PATH \    # 设置DropPath的概率，防止过拟合
#  --drop $DROP                # 设置Drop的概率，用于连接丢弃
#
#  --output $OUTPUT \
##  --experiment $EXPERIMENT  \
