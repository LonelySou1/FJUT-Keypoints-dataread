import torch
from monai.transforms import Transform
class CustomTransform(Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        # 获取标签和图像元数据
        image = data["image"]
        label = data["label"]
        size = data["image_meta_dict"]["pixdim"]
        H, W, D = label.shape
        or_size = torch.tensor([H,W,D])
        scale_factors = torch.tensor([96 / H, 96 / W, 96 / D], dtype=torch.float32)
        scaled_ls = []
        ls = []
        for i in range(1, 46):
            coordinates = torch.nonzero(label == i, as_tuple=False)
            #print(coordinates)
            try:
                mean_point = sum(coordinates) / len(coordinates)
                ls.append(mean_point)
            except:  
                ls.append(torch.tensor([-1, -1, -1]))
            
        for point in ls:
            scaled_point = point * scale_factors  # 按比例缩放坐标
            scaled_ls.append(scaled_point)

        scaled_ls = torch.stack(scaled_ls)
        
        # 返回图像、关键点和像素间距信息
        return {
            "image": image,
            "points": scaled_ls,
            "size": size[1:4],  # 返回像素间距信息
            "orig_size":or_size
        }
