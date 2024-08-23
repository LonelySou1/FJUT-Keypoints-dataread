import os
import json
import numpy as np
import math
import cv2
import PIL
import torch

from torchvision import transforms, datasets, models
from PIL import Image, ImageFont, ImageDraw

# 关键点检测19个点
# 从两个json数据集中读取坐标点数据，将他们平均化，并根据新高度和宽度调整坐标
def json_to_numpy(dataset_path,dataset_path2,new_height,new_width,h,w):
    coordinates = [] # 存储第一个数据集中的坐标点
    coordinates2 = [] # 存储第二个数据集的坐标点
    res = [] # 存储第一个数据集处理后的坐标点
    res2 = [] # 存储第二个数据集处理后的坐标点
    with open(dataset_path) as fp:  # 打开第一个文件
         for line in fp:      # 遍历文件每一行
            coordinates.append(line.strip()) # 将去除空白的行添加到第一个数据集列表
    for i in range(19):       # 循环19次，数据集有多少个坐标就循环多少次
        res.append(coordinates[i].split(','))  # 将 coordinates 中的每个字符串以逗号为分隔符拆分成列表，并添加到 res
        res[i][0] = int(res[i][0]) # 将拆分后列表的第一个元素（x坐标）转换为整数。
        res[i][1] = int(res[i][1]) # 将拆分后列表的第二个元素（y坐标）转换为整数。
    # 重复上述步骤
    with open(dataset_path2) as fp:
         for line in fp:
            coordinates2.append(line.strip())
    for i in range(19):
        res2.append(coordinates2[i].split(','))
        res2[i][0] = int(res2[i][0])
        res2[i][1] = int(res2[i][1])


    landmarks = []
    landmarks = (np.array(res) + np.array(res2)) // 2   # 将两个列表 res 和 res2 转换为NumPy数组，元素相加后除以2，以计算平均坐标点。

    
    landmarks = landmarks.reshape(-1,2)    # 将 landmarks 数组重塑为每行两个元素的格式，这通常对应于(x, y)坐标对
    yuan = landmarks.copy()
    for points in landmarks:        # 遍历每个坐标坐标
        points[1] = int(new_height * points[1] / h) # 根据新旧高度调整y坐标。
        points[0] = int(new_width * points[0] / w)  # 根据新旧宽度调整x坐标。
    return landmarks,yuan    # 返回调整后的坐标

# 关键点检测29个点
def json_to_numpy2(dataset_path,new_height,new_width):
    with open(dataset_path) as fp: # 使用`with`语句打开文件，这样可以确保文件在使用后会被正确关闭。`dataset_path`是传入的文件路径。
        json_data = json.load(fp) # 读取文件内容，并将JSON格式的字符串解析为Python字典。
        points = json_data['shapes'] # 从字典中获取键为`'shapes'`的值，通常这个值包含图像中的关键点信息。
        h, w = json_data['imageHeight'], json_data['imageWidth'] # 获取原始图像的高度和宽度。

    landmarks = [] # 初始化两个空列表和字典，用于存储关键点信息。
    landmarks_dic = {} 
    for point in points: # 循环遍历每个关键点信息。
        for p in point['points']: # 因为每个关键点可能包含多个坐标点，所以再次遍历。
            landmarks_dic[point['label']] = p # 将关键点的标签和坐标存储到字典中。

    for i in range(1,30): # 假设关键点的标签是从1到29，遍历这些标签。
        landmarks.append(landmarks_dic[str(i)])  # 将字典中的关键点坐标添加到`landmarks`列表中。

    landmarks = np.array(landmarks) # 将列表转换为NumPy数组。

    landmarks = landmarks.reshape(-1,2)   # 确保关键点的坐标是2维的（即每个关键点由x和y坐标组成）。
    for points in landmarks: # 遍历每个关键点坐标。
        points[1] = int(new_height * points[1] / h) # 根据新的图像尺寸调整关键点的坐标。
        points[0] = int(new_width * points[0] / w)
            

    return landmarks # 返回调整后的关键点坐标数组。
def json_to_numpy32point(dataset_path,new_height,new_width):
    with open(dataset_path,'r', encoding='utf-8') as fp: # 使用`with`语句打开文件，这样可以确保文件在使用后会被正确关闭。`dataset_path`是传入的文件路径。
        json_data = json.load(fp) # 读取文件内容，并将JSON格式的字符串解析为Python字典。
        points = json_data['shapes'] # 从字典中获取键为`'shapes'`的值，通常这个值包含图像中的关键点信息。
        h, w = json_data['imageHeight'], json_data['imageWidth'] # 获取原始图像的高度和宽度。

    landmarks = [] # 初始化两个空列表和字典，用于存储关键点信息。
    landmarks_dic = {} 
    for point in points: # 循环遍历每个关键点信息。
        for p in point['points']: # 因为每个关键点可能包含多个坐标点，所以再次遍历。
            landmarks_dic[point['label']] = p # 将关键点的标签和坐标存储到字典中。

    for i in range(1,33): # 假设关键点的标签是从1到29，遍历这些标签。
        landmarks.append(landmarks_dic[str(i)])  # 将字典中的关键点坐标添加到`landmarks`列表中。
    
    landmarks = np.array(landmarks) # 将列表转换为NumPy数组。
    yuan = landmarks.copy()
    landmarks = landmarks.reshape(-1,2)   # 确保关键点的坐标是2维的（即每个关键点由x和y坐标组成）。
    for points in landmarks: # 遍历每个关键点坐标。
        points[1] = int(new_height * points[1] / h) # 根据新的图像尺寸调整关键点的坐标。
        points[0] = int(new_width * points[0] / w)
            

    return landmarks,yuan # 返回调整后的关键点坐标数组。
# 根据关键点坐标生成热力图（热图）
def generate_heatmaps(landmarks,height,width,sigma,new_height,new_width): # 关键点坐标 原始图高 宽 标准差 生成新的图的高 宽
    heatmaps = [] # 存储生成的热力图
    for points in landmarks: # 遍历 landmarks 列表中的每个点（每个点都是一个坐标对，如 [x, y]）
        heatmap = np.zeros((new_height, new_width)) # 创建一个形状为 new_height x new_width 的零矩阵，这个矩阵将用作初始化的热图
        ch = int(points[1]) # 获取 y 坐标（高度），并转换为整数。
        cw = int(points[0]) # # 获取 x 坐标（高度），并转换为整数。
        
        heatmap[ch][cw] = 1 # 在热图的对应坐标点 (ch, cw) 上设置值为 1，作为高斯模糊的中心点。

        heatmap = cv2.GaussianBlur(heatmap,sigma,0) # 使用 cv2.GaussianBlur 函数应用高斯模糊。sigma 是一个元组，定义了高斯核的大小，0 是高斯核在 X 和 Y 方向的标准差（当其为 0 时，会从核大小自动计算）。
        am = np.amax(heatmap) # 计算模糊后热图中的最大值。
        heatmap /= am / 255 # 将热图的所有值规范化到 0 到 255 的范围内，使其适用于8位图像显示。
        heatmaps.append(heatmap)  # 将处理过的热图添加到 heatmaps 列表中

    heatmaps = np.array(heatmaps) # 将列表 heatmaps 转换为 NumPy 数组，方便后续处理和存储。
    return heatmaps  # 返回包含所有热图的 NumPy 数组。

# 从一批热图（heatmaps）中提取每个关节的最大预测坐标
def get_max_preds(batch_heatmaps): 

    batch_size, num_joints, h, w = batch_heatmaps.shape # 从 batch_heatmaps 的形状中提取批量大小 (batch_size)，关键点数量 (num_joints)，以及热图的高度 (h) 和宽度 (w)。
    heatmaps_reshaped = batch_heatmaps.reshape(batch_size, num_joints, -1) # 将每个热图重新塑形为一维，便于在每个热图上执行最大值操作。-1 表示自动计算该维度的大小。
    maxvals, idx = torch.max(heatmaps_reshaped, dim=2) # 在重塑后的热图中找到最大值 (maxvals) 和其对应的索引 (idx)。dim=2 表示沿着每个热图的最后一个维度（即压扁的热图）寻找最大值。

    maxvals = maxvals.unsqueeze(dim=-1) # 将 maxvals 张量在最后一个维度上增加一个维度，为后续操作准备。
    idx = idx.float() # 将 idx 张量转换为浮点数，这在后续计算中可能更方便

    preds = torch.zeros((batch_size, num_joints, 2)).to(batch_heatmaps) # 创建一个形状为 (batch_size, num_joints, 2) 的全零张量 preds，并确保它与 batch_heatmaps 位于同一个设备上（例如GPU或CPU）。

    preds[:, :, 0] = idx % w  # column 对应最大值的x坐标
    preds[:, :, 1] = torch.floor(idx / w)  # row 对应最大值的y坐标

    pred_mask = torch.gt(maxvals, 0.0).repeat(1, 1, 2).float().to(batch_heatmaps.device) # 生成一个掩码 pred_mask，其中大于 0 的 maxvals 对应的元素被设置为 True，其他为 False。通过 repeat(1, 1, 2) 将掩码复制到每个坐标上，并转换为浮点数，确保与 preds 在同一个设备上。

    preds *= pred_mask  # 将 preds 中的每个预测坐标与 pred_mask 相乘，只保留有效的坐标预测（即那些最大值大于 0 的坐标）
    return preds * 4  # 返回坐标预测，并将每个坐标乘以 4。这可能是为了缩放回原始图像的尺寸，如果热图被缩小了。

# 这个函数的输出是每个关节在原始图像尺寸上的预测坐标位置。