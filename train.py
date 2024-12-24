""" 该脚本大部分内容来自 https://github.com/rwightman/pytorch-image-models/blob/v0.6.11/train.py
并做了一些修改：
1) 启用了梯度累积（`--grad-accum-steps`）
2) 为 ConvFormer 和带有 MLP 头的 CAFormer 添加了 `--head-dropout` 参数
3) 根据 DeiT 设置了一些超参数的默认值：
-j 8 \
--opt adamw \
--epochs 300 \
--sched cosine \
--warmup-epochs 5 \
--warmup-lr 1e-6 \
--min-lr 1e-5 \
--weight-decay 0.05 \
--smoothing 0.1 \
--aa rand-m9-mstd0.5-inc1 \
--mixup 0.8 \
--cutmix 1.0 \
--remode pixel \
--reprob 0.25 \
"""

""" ImageNet 训练脚本

这是一个轻量且易于修改的 ImageNet 训练脚本，旨在通过一些最新的网络和训练技术，重现 ImageNet 训练结果。
它更倾向于使用标准的 PyTorch 和标准 Python 风格，而不是试图“做所有事情”。尽管如此，它在速度和训练结果上
相较于常规的 PyTorch 示例脚本，提供了相当多的改进。您可以根据需要重新利用此脚本。

该脚本源自 PyTorch ImageNet 示例的早期版本
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA 特定的加速来自 NVIDIA Apex 示例
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

由 / 版权所有 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils
import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, \
    LabelSmoothingCrossEntropy
# timm 是一个流行的 PyTorch 图像模型库，包含了大量的预训练模型，create_model 方法是其中的核心功能之一。

# create_model 方法用于根据指定的模型名称和配置，动态地构建并返回一个已配置好的深度学习模型。
# 该方法允许用户根据模型名称、预训练权重、输入类别数等多个参数来创建不同的神经网络模型。
# 它的设计旨在简化模型的加载和配置过程，使用户能够方便地使用和扩展各种图像分类网络。
# 作用：
#   （1）动态创建模型：通过指定模型名称，自动加载相应的模型架构
#   （2）加载预训练权重：支持加载预训练模型，这可以加速训练并提升模型性能
#   （3）灵活配置：支持多种参数，允许用户配置网络架构的各个方面，如类别数、dropout 比率、模型的初始化权重路径等
# 常见参数：
#   model_name (str): 必须参数，指定要创建的模型的名称
#   pretrained (bool): 可选参数，默认值为 False，表示是否加载预训练模型的权重
#   num_classes (int): 可选参数，指定模型的输出类别数
#   drop_rate (float): 可选参数，指定模型中某些层的 dropout 比率，用于防止过拟合
#   drop_connect_rate (float): 可选参数，指定 DropConnect 的比率。DropConnect 是一种正则化技术，通常用于深度神经网络中，旨在通过随机“丢弃”连接来提高模型的鲁棒性。
#   drop_path_rate (float): 可选参数，指定 DropPath 的比率。DropPath 是另一种类似 DropConnect 的正则化技术，通常用于更深的网络结构中
#   checkpoint_path (str): 可选参数，指定模型初始化的检查点路径。如果需要从一个已保存的检查点加载权重，则可以通过此参数提供路径
#   scriptable (bool): 可选参数，默认值为 False。如果设置为 True，则会启用 TorchScript 脚本化功能，这有助于将模型部署到非 Python 环境中
#   global_pool (str): 可选参数，指定全局池化的方式。通常是 'avg' 或 'max'，分别表示使用全局平均池化或全局最大池化
# 支持的模型：ResNet系列、EfficientNet系列、Vision Transformer（ViT）系列、Swin Transformer、其他

# convert_splitbn_model 方法用于将模型转换为支持分割批量归一化（Split BatchNorm，简称 splitbn）的版本。
# 分割批量归一化方法：将整个批次分割成多个小批次来计算每个小批次的统计量，并将这些小批次的统计量合并，从而避免了批次内样本数过少，导致统计量（均值和方差）不稳定的问题。
# 作用：修改模型的批量归一化层，使其支持分割批量归一化。
#   （1）通过将原本的标准批量归一化层转换为分割批量归一化层，使得每个小批次可以单独计算归一化参数。
#   （2）适用于大规模数据集训练，或者使用对比学习（Contrastive Learning）、分布式训练时有特殊需求的场景。
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
    convert_splitbn_model, convert_sync_batchnorm, model_parameters, set_fast_norm
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
# from timm.utils import ApexScaler, NativeScaler
from utils import ApexScalerAccum as ApexScaler
from utils import NativeScalerAccum as NativeScaler

import models

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False


torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

# 第一个参数解析器仅解析 --config 参数，该参数用于加载一个包含键值对的 YAML 文件，用于覆盖下面主解析器的默认值
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
# add_arguments是argparse模块中的方法，用于向命令行工具中添加一个新的命令行参数。作用是定义命令行参数的名称、类型、默认值以及如何处理用户的输入
# -c: 短名称
# --config: 长名称
# default属性用于设置参数的默认值。即：如果用户在命令行中没有提供该参数，该参数则使用默认值
# type属性用于设置参数的属性。即：用户提供的参数值将被解析为该类型
# metavar是帮助信息中的一个占位符，用于显示该参数期望的值的类型或描述
# help提供一个简短的描述，说明该参数的用途，在显示命令行帮助时，这个描述将作为该参数的说明
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


# 创建命令行解析器对象
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
# add_argument_group方法用于将相关的命令行参数分组，让命令行参数的组织结构更清晰，帮助用户快速了解每个参数的作用和功能
# 该方法返回一个 ArgumentGroup 对象，允许你在该分组下添加命令行参数。通过使用分组，用户能够在帮助文档中更容易地查看与数据集相关的参数。
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside of the dataset group because it is positional.
# data_dir 是作为位置参数（positional argument）的必选参数。没有前缀-或--，这意味着在命令行中，它的位置决定了它的作用
parser.add_argument('data_dir', metavar='DIR',
                    help='path to dataset')
group.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
group.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
group.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
# action='store_true'表示当用户在命令行中提供该参数时，程序将把该参数的值设为True。如果用户没有提供该参数，程序将默认将其设为False
group.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet50"')
group.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
group.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
group.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
group.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
group.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
group.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
# nargs='3'表示该参数可以接受3个值，并将这些值作为一个列表传递给程序
group.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
group.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
# nargs='+'表示该参数可以接受一个或多个值，并将这些值作为一个列表传递给程序
group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of dataset')
group.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='Input batch size for training (default: 128)')
group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                    help='Validation batch size override (default: None)')
group.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')

# add_mutually_exclusive_group方法用于将一组互斥参数添加到组中
# 互斥参数（mutually exclusive arguments）是指在命令行中，这些参数不能同时出现。如果用户指定了其中一个参数，那么其他互斥参数将无法使用
# 在这段代码中，scripting_group 是一个互斥参数组，意味着该组中的参数是互斥的，用户只能选择其中一个
scripting_group = group.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='torch.jit.script the full model')
scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
                    help="Enable AOT Autograd support. (It's recommended to use this option with `--fuser nvfuser` together)")
# --fuser 参数用于选择不同的 JIT (Just-In-Time) 编译器 fuser 类型。JIT 编译器用于在运行时将模型的计算图转换为更高效的实现，
# 而 "fuser" 是用于合并操作、优化计算图的一个选项。不同的 fuser 实现可能在性能或兼容性上有所不同，尤其是在特定硬件或环境下
group.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
group.add_argument('--fast-norm', default=False, action='store_true',
                    help='enable experimental fast-norm')
# 该选项用于控制是否启用梯度检查点功能。
# 梯度检查点是一个用于节省内存的技术，通过在反向传播时不保存某些中间梯度结果，而是重新计算这些结果来节省显存
group.add_argument('--grad-checkpointing', action='store_true', default=False,
                    help='Enable gradient checkpointing through model blocks/stages')

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
group.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
group.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
group.add_argument('--weight-decay', type=float, default=0.05,
                    help='weight decay (default: 0.05)')
group.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
group.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')
group.add_argument('--layer-decay', type=float, default=None,
                    help='layer-wise learning rate decay (default: None)')

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
group.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate (default: 0.05)')
group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
group.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                    help='amount to decay each learning rate cycle (default: 0.5)')
group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
group.add_argument('--lr-k-decay', type=float, default=1.0,
                    help='learning rate k-decay for cosine/poly (default: 1.0)')
group.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 1e-6)')
group.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
group.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--grad-accum-steps', default=1, type=int,
                    help='gradient accumulation steps')
group.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
group.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
group.add_argument('--decay-milestones', default=[30, 60], type=int, nargs='+', metavar="MILESTONES",
                    help='list of decay epoch indices for multistep lr. must be increasing')
group.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                    help='epoch interval to decay LR')
group.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
group.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
group.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
group = parser.add_argument_group('Augmentation and regularization parameters')
group.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
group.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
group.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
group.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
group.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
group.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: rand-m9-mstd0.5-inc1)'),
group.add_argument('--aug-repeats', type=float, default=0,
                    help='Number of augmentation repetitions (distributed training only) (default: 0)')
group.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
group.add_argument('--jsd-loss', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
group.add_argument('--bce-loss', action='store_true', default=False,
                    help='Enable BCE loss w/ Mixup/CutMix use.')
group.add_argument('--bce-target-thresh', type=float, default=None,
                    help='Threshold for binarizing softened BCE targets (default: None, disabled)')
group.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
group.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
group.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
group.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
group.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
group.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
group.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
group.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
group.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
group.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
group.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
group.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
group.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
group.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
group.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
group.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
group.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
group.add_argument('--head-dropout', type=float, default=0.0, metavar='PCT',
                    help='dropout rate for classifier (default: 0.0)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
# 批量归一化参数（目前仅适用于基于 gen_efficientnet 的模型）
group = parser.add_argument_group('Batch norm parameters', 'Only works with gen_efficientnet based models currently.')
group.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
group.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
# 该参数设置是否启用了同步批量归一化（Synchronized BatchNorm）。同步批量归一化用于多 GPU 分布式训练，
# 它会在所有 GPU 上同步计算均值和方差，以避免每个 GPU 上的 BN 计算不一致。
group.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
# 在每个 epoch 之后在节点之间分发批量归一化统计信息（"broadcast"、"reduce" 或 ""）
group.add_argument('--dist-bn', type=str, default='reduce',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
group.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
group = parser.add_argument_group('Model exponential moving average parameters')
group.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
group.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
group.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
group.add_argument('--worker-seeding', type=str, default='all',
                    help='worker seed mode (default: all)')
group.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
group.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
group.add_argument('-j', '--workers', type=int, default=8, metavar='N',
                    help='how many training processes to use (default: 8)')
group.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
group.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
group.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
group.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
group.add_argument('--no-ddp-bb', action='store_true', default=False,
                    help='Force broadcast buffers for native DDP to off.')
group.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
group.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
group.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
group.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
group.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
group.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
group.add_argument("--local_rank", default=0, type=int)
group.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
# wandb 是 Weights & Biases，一个常用的深度学习训练监控和可视化工具，能够记录训练过程中的超参数、日志、图表、模型以及其他指标
group.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')


def _parse_args():
    """
    解析命令行参数，支持从配置文件加载参数。首先解析可能的配置文件路径，若提供了配置文件，
    会将其中的内容作为默认参数值。然后解析剩余的命令行参数，最终返回解析后的参数对象和
    参数的 YAML 格式字符串。

    该方法的功能是：
    1. 解析命令行参数，包括从配置文件加载默认参数。
    2. 将解析后的参数对象转换为 YAML 格式的文本，便于保存或记录。

    返回：
        tuple: 返回两个值：
            - args (Namespace): 解析后的命令行参数对象。
            - args_text (str): 参数对象的 YAML 格式字符串，用于保存或记录。
    """
    # Do we have a config file to parse?
    # 使用 config_parser 解析已知的命令行参数，返回解析后的配置参数和剩余的未知参数
    args_config, remaining = config_parser.parse_known_args()
    # 检查用户是否通过命令行传递了配置文件路径（args_config.config），如果有则加载配置文件
    if args_config.config:
        # 打开指定路径的配置文件并读取内容
        with open(args_config.config, 'r') as f:
            # 使用 yaml.safe_load 解析 YAML 格式的配置文件内容，转换为字典
            cfg = yaml.safe_load(f)
            # 使用解析得到的配置字典覆盖 parser 的默认参数值
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    # 使用 parser 解析剩余的命令行参数（不包含已解析的配置参数），并返回解析后的参数对象。
    # 如果提供了配置文件，则配置文件中的默认值已经被加载并覆盖了 parser 的默认值。
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    # 将解析后的 args 对象转换为字典并使用 yaml.safe_dump 转换为 YAML 格式的文本。
    # 设置 default_flow_style=False 以确保输出的 YAML 文本是块状风格（即更易读）。
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    # 返回解析后的命令行参数对象（args）和参数的 YAML 格式字符串（args_text）。
    return args, args_text


def main():
    # 设置默认的日志配置。这通常是为了初始化日志记录器，使得程序能够记录日志信息，便于调试和追踪
    utils.setup_default_logging()
    # 解析命令行参数，返回：
    # args: 解析后的命令行参数对象
    # args_text: args 对象的 YAML 格式字符串（通常用于保存和记录日志）
    args, args_text = _parse_args()

    # 根据 args.no_prefetcher 参数的值取反设置 args.prefetcher
    args.prefetcher = not args.no_prefetcher
    # 表示默认情况下不使用分布式训练
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        # 如果 WORLD_SIZE 存在，判断其值是否大于 1。如果大于 1，说明启用了分布式训练，则将 args.distributed 设置为 True
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    # 默认设置为 'cuda:0'，表示使用第 0 个 GPU。如果程序使用多 GPU 分布式训练，后续会根据 local_rank 修改该值
    args.device = 'cuda:0'
    # 默认设置为1，表示使用 1 个进程进行训练
    args.world_size = 1
    # 设置为 0，表示全局的进程编号。分布式训练时，每个进程会有一个唯一的全局编号
    args.rank = 0  # global rank
    # 如果启用了分布式训练
    if args.distributed:
        # 检查环境变量 LOCAL_RANK 是否存在，LOCAL_RANK 通常用于表示每个进程在单个节点上的 GPU 排序
        if 'LOCAL_RANK' in os.environ:
            # 如果存在 LOCAL_RANK 环境变量，将其值解析为整数并赋值给 args.local_rank，表示当前进程在单个节点上的 GPU 排序
            args.local_rank = int(os.getenv('LOCAL_RANK'))
        # 根据 local_rank 设置当前进程使用的 GPU 设备
        args.device = 'cuda:%d' % args.local_rank
        # 设置当前进程使用的 GPU 设备
        torch.cuda.set_device(args.local_rank)
        # 初始化分布式训练环境，使用 NCCL 后端（推荐用于多 GPU 训练）。init_method='env://' 表示通过环境变量初始化进程组
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        # 获取分布式训练中的总进程数
        args.world_size = torch.distributed.get_world_size()
        # 获取当前进程的全局排名（编号）
        args.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    # 确保 args.rank 为非负数。此断言用于确保进程编号有效
    assert args.rank >= 0

    if args.rank == 0 and args.log_wandb:
        if has_wandb:
            # 初始化 wandb 记录，传入项目名称 args.experiment 和配置 args
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")

    # resolve AMP arguments based on PyTorch / Apex availability
    # 用于选择是否启用混合精度训练
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            # 如果支持 PyTorch 的原生 AMP（自动混合精度），则启用原生 AMP
            args.native_amp = True
        elif has_apex:
            # 如果支持 NVIDIA 的 APEX（加速库），则启用 APEX AMP
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    utils.random_seed(args.seed, args.rank)

    #  检查并设置 JIT 编译器 fuser
    if args.fuser:
        utils.set_jit_fuser(args.fuser)
    # 检查并设置加速归一化
    if args.fast_norm:
        # fast_norm 可能指代优化的批归一化（Batch Normalization）实现，可以加速训练过程，减少计算开销
        set_fast_norm()

    # 创建模型参数字典，将一系列与模型创建相关的参数传递给字典，这些参数从 args 获取，用于定义模型的结构和配置
    create_model_args = dict(
        model_name=args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,    # DropBlock 丢弃率，类似于 DropPath，但作用范围更大，通常是块级别的丢弃
        global_pool=args.gp,    # 是否启用全局池化（如全局平均池化）
        bn_momentum=args.bn_momentum,   # 批归一化的动量超参数，用于计算运行时均值和方差
        bn_eps=args.bn_eps, # 批归一化中的 epsilon 值，确保数值稳定性
        scriptable=args.torchscript,    # 是否启用 TorchScript，将模型脚本化，以便于后续部署
        checkpoint_path=args.initial_checkpoint
    )

    if 'convformer' in args.model or 'caformer' in args.model:
        # head_dropout 参数指定了模型头部（head）层的 dropout 比率
        create_model_args.update(head_dropout=args.head_dropout)

    # 根据参数字典构建模型
    # 通过解包字典的方式，将字典中的所有键值对作为参数传递给函数。通过这种方式，可以动态传递多个参数到函数中，以创建指定的模型
    model = create_model(**create_model_args)

    if args.num_classes is None:
        # 这行代码检查 model 对象是否具有 num_classes 属性。如果模型有该属性，则返回 True；否则返回 False，抛出异常。
        # 这个断言保证，如果用户没有在命令行中设置 num_classes，则模型本身必须定义这个属性。
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.grad_checkpointing:
        # 如果启用了梯度检查点，调用模型的 set_grad_checkpointing 方法启用该功能。
        model.set_grad_checkpointing(enable=True)

    if args.local_rank == 0:
        # m.numel() 计算每个参数张量中的元素个数
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    # 该函数用于解析并配置数据相关的设置，如数据预处理、数据集路径、批次大小等。
    # vars() 函数返回 args 对象的 __dict__ 属性，它包含了通过命令行传递的所有参数。这里将命令行参数转换为字典形式，以便传递给 resolve_data_config
    # verbose 参数用来决定是否打印详细信息
    data_config = resolve_data_config(vars(args), model=model, verbose=(args.local_rank == 0))

    # setup augmentation batch splits for contrastive loss or split bn
    # 该变量控制增强时批次的分割数量，通常用于对比学习或分割批量归一化（BatchNorm）
    num_aug_splits = 0
    if args.aug_splits > 0:
        # 断言 args.aug_splits 必须大于 1，因为将批次分割为 1 是没有意义的。
        # 批次分割的目的是为了在训练过程中进行数据增强或优化批量归一化，因此至少需要分割成两个部分
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        # 如果启用了分割批量归一化
        assert num_aug_splits > 1 or args.resplit
        # 将模型转换为支持分割批量归一化的版本
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    # 将模型移到 GPU 并根据需要启用通道优先内存布局
    # 将模型的所有参数和缓存在 GPU 上进行计算。cuda() 是 PyTorch 中用于将模型或张量移到 GPU 的方法。
    # 默认情况下，它会将模型转移到第一个可用的 GPU（如果有多个 GPU，可以指定设备）。
    model.cuda()
    # 是否使用通道优先内存布局（channels_last）
    if args.channels_last:
        # 将模型的内存布局转为 channels_last，这是 PyTorch 中的一种优化内存布局格式。它会将张量的维度顺序调整为 N, H, W, C，
        # 其中 N 是批次大小，H 和 W 是高度和宽度，C 是通道数。这种格式在一些 GPU 上的卷积操作中可以带来显著的性能提升。
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    # 为分布式训练设置同步批量归一化
    if args.distributed and args.sync_bn:
        # 禁用分布式批量归一化（dist_bn）设置。当启用同步批量归一化时，不再使用传统的分布式批量归一化（dist_bn）方法。
        args.dist_bn = ''  # disable dist_bn when sync BN active
        # 断言 args.split_bn 不为 True。这确保了不能同时启用分割批量归一化（Split BatchNorm）和同步批量归一化（SyncBN），因为这两种方法不兼容。
        assert not args.split_bn
        if has_apex and use_amp == 'apex':
            # Apex SyncBN used with Apex AMP
            # WARNING this won't currently work with models using BatchNormAct2d
            # 如果使用了 Apex，则调用以下函数将模型转换为支持同步批量归一化的版本
            model = convert_syncbn_model(model)
        else:
            # 如果没有使用 Apex，则使用默认的 PyTorch 函数 convert_sync_batchnorm 将模型转换为支持同步批量归一化的版本
            model = convert_sync_batchnorm(model)
        if args.local_rank == 0:
            # local_rank 是分布式训练中的一个常见变量，用来表示当前进程的 ID，local_rank == 0 表示这是第一个进程（主进程）
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)
    if args.aot_autograd:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')


    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = utils.ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank], broadcast_buffers=not args.no_ddp_bb)
        # NOTE: EMA model does not need to be wrapped by DDP

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    # create the train and eval datasets
    dataset_train = create_dataset(
        args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        repeats=args.epoch_repeats)
    dataset_eval = create_dataset(
        args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size)

    total_batch_size = args.batch_size * args.grad_accum_steps * args.world_size
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    if args.local_rank == 0:
        _logger.info('Total batch size: {}'.format(total_batch_size))

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = utils.get_outdir(args.output if args.output else './output/train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = utils.CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn,
                grad_accum_steps=args.grad_accum_steps, num_training_steps_per_epoch=num_training_steps_per_epoch
                )

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                utils.distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    utils.distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                ema_eval_metrics = validate(
                    model_ema.module, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if output_dir is not None:
                utils.update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None, log_wandb=args.log_wandb and has_wandb)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None,
        grad_accum_steps=1, num_training_steps_per_epoch=None):

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()

    model.train()
    optimizer.zero_grad()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        step = batch_idx // grad_accum_steps
        if step >= num_training_steps_per_epoch:
            continue
        # last_batch = batch_idx == last_idx
        last_batch = ((batch_idx + 1) // grad_accum_steps) == num_training_steps_per_epoch
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            output = model(input)
            loss = loss_fn(output, target)

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))


        update_grad = (batch_idx + 1) % grad_accum_steps == 0
        loss_update = loss / grad_accum_steps
        if loss_scaler is not None:
            loss_scaler(
                loss_update, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order, update_grad=update_grad)
        else:
            loss_update.backward(create_graph=second_order)
            if update_grad:
                if args.clip_grad is not None:
                    utils.dispatch_clip_grad(
                        model_parameters(model, exclude_head='agc' in args.clip_mode),
                        value=args.clip_grad, mode=args.clip_mode)
                optimizer.step()

        if update_grad:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.local_rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                acc1 = utils.reduce_tensor(acc1, args.world_size)
                acc5 = utils.reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


if __name__ == '__main__':
    main()
