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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:
    # 高斯模型实例，用于处理场景中的高斯模型数据
    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        初始化场景对象，加载场景数据，处理相机配置和高斯模型初始化。
        :param args: ModelParams对象，包含模型和场景的配置参数
        :param gaussians: 高斯模型对象，用于管理场景中的高斯数据
        :param load_iteration: 指定加载场景数据的迭代次数，若未指定，则自动寻找最大迭代次数
        :param shuffle: 是否随机打乱相机顺序，用于训练数据的随机化
        :param resolution_scales: 相机渲染的分辨率比例列表，支持多分辨率处理
        """
        self.model_path = args.model_path  # 模型存储路径
        self.loaded_iter = None  # 已加载的迭代次数
        self.gaussians = gaussians  # 高斯模型对象

        # 根据指定迭代加载训练模型，或搜索最大迭代次数
        if load_iteration:
            if load_iteration == -1:
                # 搜索最大迭代文件夹
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("加载训练模型于迭代 {}".format(self.loaded_iter))

        self.train_cameras = {}  # 存储训练用相机配置
        self.test_cameras = {}  # 存储测试用相机配置

        # 检测场景类型并加载相关场景信息
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            # 加载Colmap格式的场景数据
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            # print("发现 transforms_train.json 文件，假定 Blender 数据集!")
            # 加载Blender格式的场景数据
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            # assert False, "无法识别场景类型!"
            assert False, "Could not recognize scene type!"

        # 初始化场景的点云和相机信息
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # 如果启用shuffle，对训练和测试相机列表进行随机洗牌
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 根据分辨率比例加载训练和测试相机配置
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        # 根据加载的迭代次数加载或创建高斯模型
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        """
        保存当前场景的点云到指定迭代文件夹
        :param iteration: 指定的迭代次数
        """
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        """
        获取指定分辨率比例的训练相机列表
        :param scale: 相机渲染的分辨率比例
        :return: 训练相机列表
        """
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        """
        获取指定分辨率比例的测试相机列表
        :param scale: 相机渲染的分辨率比例
        :return: 测试相机列表
        """
        return self.test_cameras[scale]