import torch
from monai.transforms import Transform
import json
from my_predataset import json_to_numpy, generate_heatmaps,json_to_numpy2
import numpy as np
import torch


class ReadPoint19point(Transform):
    def __init__(self,size = 672):
        super().__init__()
        self.size = size
    def __call__(self, data):
        # 初始化一个空的 32x2 的数组，用于存储点坐标
        
        points_array = np.zeros((32, 2))
        # 读取 JSON 文件
        with open(data['label1'], 'r', encoding='utf-8') as file:
            lines = file.readlines()
        coordinates1 = []

        for i,line in enumerate(lines):
            if i<19:
                x, y = map(int, line.split(','))
                coordinates1.append([x, y])
            else:
                break

        with open(data['label2'], 'r', encoding='utf-8') as file:
            lines = file.readlines()
        coordinates2 = []

        for i,line in enumerate(lines):
            if i<19:
                x, y = map(int, line.split(','))
                coordinates2.append([x, y])
            else:
                break
        points_array = np.array([[(x1 + x2)/2, (y1 + y2)/2] for (x1, y1), (x2, y2) in zip(coordinates1, coordinates2)])
        
        scales = np.array([self.size/1935,self.size/2400])
        points_array = points_array*scales
        points_array = points_array.astype(int)
        
        
        targets = generate_heatmaps(points_array,self.size,self.size,(55,55),
                                     new_height=self.size,new_width=self.size)
        targets = torch.tensor(targets, dtype=torch.float32)  
        # heatmaps = torch.tensor(heatmaps, dtype=torch.float32) 
        # print(heatmaps.dtype)
        return {


            "image":data['image'],
            'heatmaps':targets,
            "points":points_array,
        } 

