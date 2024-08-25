import json
import os
import time
import torch
from torch.utils import data
import numpy as np
import sys
from loss import KpLoss
import distributed_utils as utils
import numpy as np
import nibabel as nib
from monai.transforms import (
    LoadImaged,
    LoadImage,
    AddChanneld,
    ToTensord,
    Compose,
    MapTransform,
    Resized,
    SqueezeDimd,
    EnsureChannelFirstd,
    Rotate90d,
    Flipd
)
from MyTransform import ReadPoint19point
from monai.data import DataLoader, Dataset, pad_list_data_collate, CacheDataset,PersistentDataset
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.optimizers import Novograd
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import math
import matplotlib.pyplot as plt
from model import UNetHao3,UNetEGA3,UNetHao4
# 参与训练
result = [] # 存储验证结果
def euclidean_distance(tensor1, tensor2, dim, device):
   
    # 计算差值的平方
    squared_difference = (tensor1.to(device) - tensor2.to(device))**2
    
    # 沿着指定的维度求和 得到每一组差的平方和。
    sum_along_dim = squared_difference.sum(dim)
    
    # 取平方根 得到欧几里得距离
    distance = torch.sqrt(sum_along_dim)
    
    # 得到张量距离
    return distance
def get_max_preds(batch_heatmaps):

    batch_size, num_joints, h, w = batch_heatmaps.shape
    heatmaps_reshaped = batch_heatmaps.reshape(batch_size, num_joints, -1)
    maxvals, idx = torch.max(heatmaps_reshaped, dim=2)

    maxvals = maxvals.unsqueeze(dim=-1)
    idx = idx.float()

    preds = torch.zeros((batch_size, num_joints, 2)).to(batch_heatmaps)

    preds[:, :, 0] = idx % w  # column 对应最大值的x坐标
    preds[:, :, 1] = torch.floor(idx / w)  # row 对应最大值的y坐标

    pred_mask = torch.gt(maxvals, 0.0).repeat(1, 1, 2).float().to(batch_heatmaps.device)

    preds *= pred_mask

    return preds 
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
    loss = KpLoss()
    
    # 初始化均值损失
    mloss = torch.zeros(1).to(device)  

    # 遍历数据加载器，并记录每次迭代
    for i, item in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # images = torch.stack([image.to(device) for image in images]): 将图像数据转移到设备（如GPU）并堆叠
        images, targets, points= item['image'].to(device), item['heatmaps'].to(device),item['points']
        # 转换为 torch.Tensor
        
        images = torch.stack([image.to(device) for image in images])
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            
            results = model(images) # 执行模型前向传播
            losses = loss(results, targets) # 计算关键点损失 均值方差  热力图关键点和真实关键点标签
        
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
    batch_size = args.batch_size
    def is_valid_file(filename, extensions):
        return any(filename.endswith(ext) for ext in extensions)

    print("Using {} device training.".format(device.type)) # 打印设备类型
    tdata_dir = "/home/hliang/Key/data2/train"
    data_dir = "/home/hliang/Key/data2/test1"
    image_extensions = [ ".png", ".jpg", ".bmp"]
    t_images = sorted([os.path.join(tdata_dir, f) for f in os.listdir(tdata_dir) if is_valid_file(f, image_extensions)])
    t_labels = sorted([os.path.join(tdata_dir, f) for f in os.listdir(tdata_dir) if f.endswith(".txt")])
    tlabels1 = [t_labels[i] for i in range(len(t_labels)) if i % 2 == 0]
    tlabels2 = [t_labels[i] for i in range(len(t_labels)) if i % 2 == 1]
    train_files = [{"image": img, "label1": lbl1, "label2":lbl2} for img, lbl1, lbl2 in zip(t_images, tlabels1, tlabels2)]
    
    images = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if is_valid_file(f, image_extensions)])
    labels = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")])
    vlabels1 = [labels[i] for i in range(len(labels)) if i % 2 == 0]
    vlabels2 = [labels[i] for i in range(len(labels)) if i % 2 == 1]
    val_files = [{"image": img, "label1": lbl1, "label2": lbl2} for img, lbl1, lbl2 in zip(images, vlabels1, vlabels2)]

    # for i in val_files:
    #     print(i["image"])
    #     print(i['label1'])
    #     print(i["label2"])
    #     print("------------")
    image_size = 672
    train_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]), 
        Resized(keys=['image'], spatial_size=[672, 672], mode='bilinear', align_corners=True),
        Rotate90d(keys=['image'],k=3),
        Flipd(keys=["image"], spatial_axis=1),
        ReadPoint19point(Gaoh=55,size=image_size),
    ])
    cache_dir1 = "/home/hliang/Key/datasave"
    train_ds = CacheDataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    val_ds = CacheDataset(data=val_files, transform=train_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=4)
    loss = KpLoss()
    # 创建模型
    model = UNetHao4(3,19) 
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
        L = []
        # 日志记录器
        metric_logger = utils.MetricLogger(delimiter="  ")
        for i, item in enumerate(metric_logger.log_every(val_loader, 50)):
        # images = torch.stack([image.to(device) for image in images]): 将图像数据转移到设备（如GPU）并堆叠
            images, targets, points= item['image'].to(device), item['heatmaps'].to(device),item['points']
            images = torch.stack([image.to(device) for image in images])
            # 输入模型得到热力图
            results = model(images)
            # results, results2 = model(images)
            # 得到预测点
            point1 = get_max_preds(results)[0].to(device)

            # point11 = get_max_preds(results2)[0].to(device)
            # point1 = (point1 + 2 * point11) / 2
            # point2 = get_max_preds(targets)[0].to(device)
            # 对关键点进行缩放 使他们与图像的实际尺寸相对应 高度和宽度的缩放比例 y是缩放因子
            y = torch.tensor([193.5 / image_size,240 / image_size]).to(device) # 与关键点热力图对应，但是要多加个0，因为1935X2400是0.1毫米

            point1 = point1 * y  # 将热力图坐标尺度转换到原始图像尺度

            point2 = points[0].to(device) # 将当前坐标放到设备上
            point2 = point2 * y  # 坐标进行缩放
            # point2 = point2 * y
            # print(point1.size())[29,2]
            # print(point2.size())[29,2]
            # 计算预测坐标与真实坐标的距离   欧式距离   评估模型的准确性
            y = euclidean_distance(point1,point2,1,device) 
            # 所有关键点的欧氏距离求和 然后除以关键点的数量。 为了求平均误差   平均欧式距离
            y2 = y.sum(dim=0) / 19.0
            # 记录了每个验证批次的平均欧式距离，用于进一步分析或评估模型的整体性能。
            L.append(y2)

        l = 0 # 存储平均欧式距离之和
        for i in L:
            l += i
        # item 张量转化为数值  150个验证批次，得到每个批次的平均SDR  得到平均误差
        print('SDR=',l.item() / 150.0)   # 直观的看到模型在验证集的性能
        # SDR值存放到列表  方便后续操作
        result.append(l.item() / 150.0)

        # 每个训练周期结束后，保存模型的状态   模型擦拭农户，优化器状态，学习率   训练周期信息
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        # 是否使用混合精度 保存AMP的缩放器状态
        if args.amp:
            save_files["scaler"] = scaler.state_dict()

        # 保存的文件名包含当前的 SDR 值
        torch.save(save_files, "/data/hliang/weight/Unet1024/672model-{}.pth".format(l.item() / 150.0))
        print("MRE:",sum(L)/len(L))
        end_time = time.time()
    
    # 计算一个epoch所用的时间
        epoch_time = end_time - start_time
        
        print(f"Epoch [{epoch}] completed in {epoch_time:.2f} seconds.")
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:1', help='device')
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
