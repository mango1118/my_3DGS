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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        # 从缩放旋转因子里构建协方差矩阵
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            # 旋转乘缩放，得到高斯椭球的变化，得到L矩阵
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            # L乘L第一维和第二维的转置，第零维是高斯数量，所以跳过
            # 构建出真实的协方差矩阵
            actual_covariance = L @ L.transpose(1, 2)
            # 只保留上三角，因为是对称矩阵
            symm = strip_symmetric(actual_covariance)
            return symm

        # 定义激活函数，缩放因子的激活函数就是exp
        self.scaling_activation = torch.exp
        # 对应缩放因子的反激活函数是log
        self.scaling_inverse_activation = torch.log
        # 协方差矩阵没用激活函数，因为旋转和缩放都激活过了，所以直接用刚才的方法构造
        self.covariance_activation = build_covariance_from_scaling_rotation
        # 不透明度的激活函数用sigmoid，为了不透明度在0-1之间
        self.opacity_activation = torch.sigmoid
        # 对应的反函数就是反激活函数
        self.inverse_opacity_activation = inverse_sigmoid
        # 旋转操作的激活函数是归一化函数
        self.rotation_activation = torch.nn.functional.normalize


    # 对变量进行初始化，设置成0或者空
    def __init__(self, sh_degree : int):
        # 球谐函数的阶数
        self.active_sh_degree = 0
        # 球谐函数的最高阶数是传进来的
        self.max_sh_degree = sh_degree
        # 椭球位置
        self._xyz = torch.empty(0)
        # 球谐函数的直流分量
        self._features_dc = torch.empty(0)
        # 球谐函数的高阶分量
        self._features_rest = torch.empty(0)
        # 缩放因子
        self._scaling = torch.empty(0)
        # 旋转因子
        self._rotation = torch.empty(0)
        # 不透明度
        self._opacity = torch.empty(0)
        # 投影到平面后的二维高斯分布的最大半径
        self.max_radii2D = torch.empty(0)
        # 点位置的梯度累积值
        self.xyz_gradient_accum = torch.empty(0)
        # 统计的分母数量，梯度累积值需要除以分母数量来计算每个高斯分布的平均梯度
        self.denom = torch.empty(0)
        # 优化器
        self.optimizer = None
        # 百分比目的，做密度控制
        self.percent_dense = 0
        # 学习率因子
        self.spatial_lr_scale = 0
        # 创建激活函数的方法
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    # 获取变量时，返回的是激活后的变量，所以需要用反激活函数来把变量提取出来
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    # 迭代球谐函数的阶数，如果球谐函数的阶数小于规定的最大阶数，运行这个方法之后阶数就会增加
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # 从点云文件中创建数据
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float): # 对象，传入点云，学习率的变化因子
        # 将变化因子传入对象的学习率
        self.spatial_lr_scale = spatial_lr_scale
        # 创建张量来保存点云数据，把数组类型的点云数据存到张量里
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # 把点云颜色存到RGB张量里面，转成球谐函数的系数
        # 这里只存了零阶的，也就是直流分量的球谐函数，其他后面再加上
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        # 定义了张量，维度是高斯分布总数，3个通道，球谐函数的系数数量
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        # 第0阶就是直流分量
        features[:, :3, 0 ] = fused_color
        # 高阶先定义为0，默认点云的点是只有点的颜色
        features[:, 3:, 1:] = 0.0

        # 打印初始化的点的数量
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # 计算distance，首先根据点云创建一个张量，设置最小距离0.0000001，distCUDA2函数在simple-knn里，计算对应点云最近邻居
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # 高斯球的半径就是到最近的三个高斯点的距离的平均值，有最小距离因此不会有重合的高斯椭球
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        # 创建旋转变量，维度是N*4的张量
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        # 存四位数w，x，y，z，将w设成零，则2arcosw就是0，其他xyz也是0，单位四元数的整体值就是0，即将旋转因子初始化为0
        rots[:, 0] = 1

        # 初试不透明度
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # 高斯椭球的分布位置用参数存储，规定梯度将来进行优化
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        # 球谐函数的直流分量，规定梯度将来进行优化
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # 球谐函数的高阶分量，规定梯度将来进行优化
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        # 旋转
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        # 缩放
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        # 不透明度
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # 二维投影分布的高斯最大半径
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        # 几种参数的传递
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 动态的学习率存储，每个参数的学习率都不一样
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        # Adam优化器
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    # 更新学习率
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            # 如果识别到参数是xyz，对学习率进行优化，优化后的学习率传回参数
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    # 从参数里面创建列表
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    # 结果保存到点云文件
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # 重置不透明度
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    # 加载点云文件
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    # 将张量转到优化器里，用于优化
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                # 参数发生大的变化，但优化器里存的动量应该是不变的，保证原有的状态不丢失，最后损失下降就是一个平滑的下降状态
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # 重置优化器
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                # 要保留哪些状态
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # 重置点，在做高斯点的修剪，不需要的高斯点就用mask删掉
    def prune_points(self, mask):
        # 选择要保留哪些点
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    # 创建新的张量并存到优化器里
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    # 给自适应密度添加新的高斯点要用到的函数
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        # 一堆属性，要添加的属性赋值给这些属性，
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        # 调用创建优化器的方法，将这些属性添加到优化器里
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        # 把新的值赋给新的对象，创造新的高斯点
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    # 自适应密度的分裂操作，参数有目前优化过程中的梯度、设定的梯度阈值、场景范围、特定常数2
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        # 先读取高斯分布的总数
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        # 创建一个全0的张量，大小就是点的高斯分布的总数的大小，用这个变量存储每个高斯分布现在的梯度
        padded_grad = torch.zeros((n_init_points), device="cuda")
        # 根据这些梯度扩展一个维度
        padded_grad[:grads.shape[0]] = grads.squeeze()
        # 然后生成掩码，如果梯度大于给定梯度的阈值，就根据掩码做一个标记
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        # 再进行一次筛选，对于这个高斯分布的缩放因子中，最大的一个维度的值大于场景的范围乘以一个比例因子，（就是高斯分布的大小已经大于要求的场景范围）
        # 这个高斯就要进行分裂
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        # 新的高斯分布的标准差取自于原本高斯分布的标准差，然后把他扩展成两个
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        # 新高斯分布的位置规定到全0的位置
        means =torch.zeros((stds.size(0), 3),device="cuda")
        # 新高斯分布就通过原本的均值和标准差进行设计
        samples = torch.normal(mean=means, std=stds)
        # 创建旋转矩阵，也是根据需要分裂的高斯进行创建
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        # 用创建好的高斯分布，先增加一个维度，然后缩放因子跟旋转矩阵相乘，得到协方差矩阵，然后再删掉新增的维度
        # 加上原本高斯分布的位置，就得到新高斯分布的位置
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        # 获取新的缩放因子之后，除以0.8*2，也就是原文中的1.6，两个变小成为小的高斯分布
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        # 旋转矩阵，球谐函数，不透明度都调用原本的
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        # 把新的变量添加到高斯分布里
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        # 创建一个过滤器，把之前的一些高斯分布删掉
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        # 清除掉中间变量
        self.prune_points(prune_filter)

    # 克隆
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        # 高斯分布的梯度的阈值大于设定的梯度阈值，就要进行克隆，判断形状是否小于场景设定的形状范围
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # 满足条件就要标记为需要克隆
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # 原高斯分布的所有变量都添加到新变量里，增加一个新高斯
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    # 高斯椭球的剔除
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        # 梯度的累加值除以分母，得到平均梯度
        grads = self.xyz_gradient_accum / self.denom
        # 归零操作
        grads[grads.isnan()] = 0.0

        # 进行分裂和克隆
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # 如果高斯椭球的不透明度小于设定的最低不透明度，就标记出来
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            # 二维高斯分布的半径大于最大的屏幕尺寸，也标记
            big_points_vs = self.max_radii2D > max_screen_size
            # 高斯分布的大小大于场景范围*0.1，也标记
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            # 为这些高斯椭球设置一个掩码，同一剔除
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        # 清除缓存
        torch.cuda.empty_cache()

    # 添加自适应密度控制过程中的状态，就是记录需要累加的梯度
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # 累计梯度，把x和y方向上的梯度给标记起来，添加到对应的计数器
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        # 每处理一个点，就给分母+1
        self.denom[update_filter] += 1