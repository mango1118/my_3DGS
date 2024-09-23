#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
# 三维空间到像平面的转换过程
class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        # 相机UID，可能有几个相机视角会重合，所以用uid区分
        self.uid = uid
        # colmap算出的相机位姿
        self.colmap_id = colmap_id
        # 旋转矩阵
        self.R = R
        # 平移矩阵
        self.T = T
        # 水平方向的视野范围（角度）
        self.FoVx = FoVx
        # 垂直方向的视野范围（角度）
        self.FoVy = FoVy
        # 图像名字
        self.image_name = image_name

        # 多卡处理
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # 图像的大小转成归一化的值，放到设备上
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        # 图像的宽和高设置到一个新的变量里
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        # 判断有没有指定alpha掩码（不知道有什么用）
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        # 相机的近点和远点
        self.zfar = 100.0
        self.znear = 0.01

        # 平移跟缩放转换的值
        self.trans = trans
        self.scale = scale

        # 世界坐标系到相机坐标系的转化，可以看getworld2view2方法，求转置再给变量
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        # 需要把相机坐标系上的高斯分布再投影到一个二维平面里，先投影到归一化的坐标系里，看函数
        # 返回的P矩阵可以把相机坐标系中的点转到一个归一化的NDC坐标系中，然后再投影到像平面做光栅化处理
        # 投影到像平面的代码在cuda里
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        # 世界坐标系到相机坐标系的投影矩阵乘相机坐标系到NDC归一化坐标系的矩阵，就得到了世界坐标系到归一化坐标系的投影矩阵
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # 得到相机光心
        self.camera_center = self.world_view_transform.inverse()[3, :3]

# 没啥用
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

