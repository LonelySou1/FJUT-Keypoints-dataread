import json
import os
import time
import torch
from torch.utils import data
import numpy as np
import sys
from Mydataset import CustomDataset,CustomDataset1,CustomDataset2
import distributed_utils as utils
import numpy as np
import nibabel as nib
from monai.transforms import (
    LoadImaged,
    AddChanneld,
    ToTensord,
    Compose,
    MapTransform,
    Resized,
    SqueezeDimd,
)
from loss import MRELoss
from monai.data import DataLoader, Dataset, pad_list_data_collate, CacheDataset,PersistentDataset
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.optimizers import Novograd
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import math
from model import UNet3D,UnetHao,UNetEGA3D,UNetEGA3D2
import matplotlib.pyplot as plt
from MyTransform import CustomTransform,CustomTransform0
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 参与训练
result = [] # 存储验证结果

LOS = []    # 存储验证损失

# 一个训练周期 模型  数据
def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=100, warmup=False, scaler=None):
    # 训练模式 开始训练
    model.train()
    # 日志
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    # 开始预热
    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    
    # 初始化关键点损失函数  均值损失
    mse = MRELoss()
    
    # 初始化均值损失
    mloss = torch.zeros(1).to(device)  

    # 遍历数据加载器，并记录每次迭代
    for i, item in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # images = torch.stack([image.to(device) for image in images]): 将图像数据转移到设备（如GPU）并堆叠
        images, labels, pixsize ,orig_size= item['image'].to(device), item['points'].to(device),item['size'],item['orig_size']
        #print(pixsize,orig_size)
        images = torch.stack([image.to(device) for image in images])
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            results = model(images) # 执行模型前向传播

            losses = mse(results, labels) # 计算关键点损失 均值方差  热力图关键点和真实关键点标签
            #print(losses)
        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = utils.reduce_dict({"losses": losses}) # 在所有GPU上简化损失数据。
        losses_reduced = sum(loss for loss in loss_dict_reduced.values()) # 计算简化后的总损失。
        
        # 获取损失值
        loss_value = losses_reduced.item()

        # 更新平均损失
        mloss = (mloss * i + loss_value) / (i + 1)

        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # 重置优化器梯度
        optimizer.zero_grad()
        # 混合精度
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)
    # 记录损失    
    LOS.append(mloss)
    # 关键点平均损失 和 当前学习率
    return mloss, now_lr




def main(args): # 传递配置文件
    start_time = time.time()
    # 选择设备
    # device = torch.device("cpu")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type)) # 打印设备类型
    
    batch_size = args.batch_size
    tdata_dir = "/data/hliang/3DCPCTnomiss/train"
    data_dir = "/data/hliang/3DCPCTnomiss/test"
    t_images = sorted([os.path.join(tdata_dir, f) for f in os.listdir(tdata_dir) if f.endswith("-image.nii.gz")])
    t_labels = sorted([os.path.join(tdata_dir, f) for f in os.listdir(tdata_dir) if f.endswith("-label.nii.gz")])
    train_files = [{"image": img, "label": lbl} for img, lbl in zip(t_images, t_labels)]
    images = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith("-image.nii.gz")])
    labels = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith("-label.nii.gz")])
    val_files = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]


    image_size = 96
    train_transforms = Compose([
        LoadImaged(keys=['image','label']),
        AddChanneld(keys=['image','label']),  
        Resized(keys=['image'], spatial_size=[image_size, image_size, image_size], mode='trilinear',align_corners=True),  # 对图像进行三线性插值
        SqueezeDimd(keys=['label'], dim=0),
        ToTensord(keys=['image', 'label']),
        CustomTransform0(size = image_size),
    ])
    cache_dir1 = "/data/hliang/3DCPCTnomiss/trainP0"
    cache_dir2 = "/data/hliang/3DCPCTnomiss/testP0"
    train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir=cache_dir1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_ds = PersistentDataset(data=val_files, transform=train_transforms, cache_dir=cache_dir2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=4)
    loss = MRELoss()
    # 创建模型
    model = UNetEGA3D2(1,45) 
    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print("the training process from epoch{}...".format(args.start_epoch))
    

    # 在这里记录损失
    train_loss = [] # 记录训练损失 
    learning_rate = [] # 学习率
    val_map = []  # MAP（平均精度）的列表

    # 训练300个epoch
    for epoch in range(args.start_epoch, args.epochs):
        
        # 开始训练 每50次迭代打印一次信息，并且记录平均损失和学习率
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader,
                                              device=device, epoch=epoch,
                                              print_freq=50, warmup=True,
                                              scaler=scaler)
        # 记录平均损失
        train_loss.append(mean_loss.item())
        # 当前学习率
        learning_rate.append(lr)

        # 更新学习率
        lr_scheduler.step()
      
        # 开始评估模式
        model.eval()
        
        # 记录指标
        L = 0
        # 日志记录器
        metric_logger = utils.MetricLogger(delimiter="  ")
        for i, item in enumerate(metric_logger.log_every(val_loader,50)):

            images, labels, pixsize ,orig_size= item['image'].to(device), item['points'].to(device),item['size'].to(device),item['orig_size'].to(device)
            images = torch.stack([image.to(device) for image in images])
            results = model(images)
            scale = orig_size/96
            label1 = labels * scale * pixsize
            predict = results * scale * pixsize
            
            bj = torch.zeros(3).to('cuda:0')
            rpoint1 = []
            rpoint2 = []
            for i, (p1, p2) in enumerate(zip(label1[0], predict[0])):
                if (p1.to(device) == bj.to(device)).all():
                    continue
                rpoint1.append(p1)
                rpoint2.append(p2)
            rpoint1 = torch.stack(rpoint1)
            rpoint2 = torch.stack(rpoint2)


            ls = loss(rpoint1, rpoint2)

            L = L + float(ls)

        print("MRE:",L/len(val_loader))
        end_time = time.time()
    
    # 计算一个epoch所用的时间
        epoch_time = end_time - start_time
        
        print(f"Epoch [{epoch}] completed in {epoch_time:.2f} seconds.")
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录(coco2017)
    parser.add_argument('--data-path', default='/home/hliang/3DKeyPoint/data/wy', help='dataset')
    # COCO数据集人体关键点信息

    # 原项目提供的验证集person检测信息，如果要使用GT信息，直接将该参数置为None，建议设置成None
    parser.add_argument('--fixed-size', default=[672, 672], nargs='+', type=int, help='input size')
    # keypoints点数
    parser.add_argument('--num-joints', default=32, type=int, help='num_joints')
    # 文件保存地址
    parser.add_argument('--output-dir', default='/home/hliang/3DKeyPoint/weight', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-steps', default=[170, 200], nargs='+', type=int, help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    # 学习率
    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # AdamW的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 训练的batch size
    parser.add_argument('--batch-size', default=4, type=int, metavar='N',
                        help='batch size when training.')
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
