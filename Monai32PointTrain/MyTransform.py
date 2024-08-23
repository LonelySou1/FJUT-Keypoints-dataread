import torch
from monai.transforms import Transform
import json
from my_predataset import json_to_numpy, generate_heatmaps,json_to_numpy2
import numpy as np
import torch


class ReadPoint32point(Transform):
    def __init__(self,size = 672):
        super().__init__()
        self.size = size
    def __call__(self, data):
        # 初始化一个空的 32x2 的数组，用于存储点坐标
        points_array = np.zeros((32, 2))
        # 读取 JSON 文件
        with open(data['label'], 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        shapes = json_data.get("shapes", [])

        for shape in shapes:
            label = shape.get("label")
            
            # 确保 label 是数字，并且在 1 到 32 之间
            if label.isdigit():
                index = int(label) - 1  # 将 label 转换为索引（0-31）
                if 0 <= index < 32:
                    points = shape.get("points", [[0, 0]])
                    points_array[index] = points[0]  # 存入点的坐标
        scales = [self.size/json_data['imageWidth'],self.size/json_data['imageHeight']]
        points_array = points_array*scales
        points_array = points_array.astype(int)
        points_array = torch.tensor(points_array, dtype=torch.float32)


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

