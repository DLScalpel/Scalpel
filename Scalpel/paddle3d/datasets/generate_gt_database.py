# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import os.path as osp
import pickle
from collections import defaultdict
from typing import List
import warnings

from paddle3d.apis.config import Config
from paddle3d.datasets.kitti.kitti_pointcloud_det import KittiPCDataset
from paddle3d.datasets.apollo.apollo_pointcloud_det import \
    ApolloPCDataset
from paddle3d.datasets.apollo.apollo_utils import class_information
from paddle3d.datasets.nuscenes.nuscenes_pointcloud_det import \
    NuscenesPCDataset
from paddle3d.geometries.bbox import get_mask_of_points_in_bboxes3d
from paddle3d.transforms.reader import (LoadPointCloud,
                                        RemoveCameraInvisiblePointsKITTI)
from paddle3d.transforms.transform import FilterSmallBBox
from paddle3d.utils.logger import logger


def generate_kitti_gt_database(dataset_root: str,
                               save_dir: str = None,
                               load_point_dim: int = 4,
                               use_point_dim: int = 4):
    if save_dir is None:
        save_dir = dataset_root

    save_dir = osp.join(save_dir, "kitti_train_gt_database")

    transforms = [
        LoadPointCloud(dim=load_point_dim, use_dim=use_point_dim),
        RemoveCameraInvisiblePointsKITTI()
    ]

    dataset = KittiPCDataset(
        dataset_root=dataset_root, mode="train", transforms=transforms)

    database = defaultdict(list)
    msg = "Begin to generate a database for the kitti_dataset_root dataset."
    cls_names = dataset.class_names

    for data_idx in logger.range(len(dataset), msg=msg):
        sample = dataset[data_idx]
        image_idx = int(sample.meta.id)
        points = sample.data
        bboxes_3d = sample.bboxes_3d
        labels = sample.labels
        difficulties = sample.difficulties

        num_bboxes = len(bboxes_3d)
        if num_bboxes == 0:
            continue
        masks = get_mask_of_points_in_bboxes3d(
            points, bboxes_3d)  # mask shape: [num_points, num_bboxes]
        for box_idx in range(num_bboxes):
            cls_name = cls_names[labels[box_idx]]
            if cls_name.lower() == "dontcare":
                continue
            mask = masks[:, box_idx]
            selected_points = points[mask]
            selected_points[:, 0:3] -= bboxes_3d[box_idx, 0:3]

            if not osp.exists(osp.join(save_dir, cls_name)):
                os.makedirs(osp.join(save_dir, cls_name))
            lidar_file = osp.join(
                osp.join(save_dir, cls_name), "{}_{}_{}.bin".format(
                    image_idx, cls_name, box_idx))

            with open(lidar_file, "w") as f:
                selected_points.tofile(f)

            anno_info = {
                "lidar_file":
                osp.join("kitti_train_gt_database", cls_name,
                         "{}_{}_{}.bin".format(image_idx, cls_name, box_idx)),
                "cls_name":
                cls_name,
                "bbox_3d":
                bboxes_3d[box_idx, :],
                "box_idx":
                box_idx,
                "data_idx":
                image_idx,
                "num_points_in_box":
                selected_points.shape[0],
                "lidar_dim":
                use_point_dim,
                "difficulty":
                difficulties[box_idx]
            }
            database[cls_name].append(anno_info)

    db_anno_file = osp.join(osp.join(save_dir, 'anno_info_train.pkl'))
    with open(db_anno_file, 'wb') as f:
        pickle.dump(database, f)
    logger.info("The database generation has been done.")

def generate_apollo_gt_database(config: str):
    cfg = Config(path=config)
    cfg = cfg.dic['train_dataset']['transforms']
    load_point_dim = cfg[0]['dim']
    use_point_dim = cfg[0]['use_dim']
    sep = cfg[0]['sep']
    dataset_root = cfg[1]['database_root']
    dataset_list = cfg[1]['database_anno_list']
    cls_names = cfg[1]['class_names']

    for dataset_name in dataset_list:
        save_dir = osp.join(dataset_root, dataset_name, "apollo_train_gt_database")

        transforms = [
            LoadPointCloud(dim=load_point_dim, use_dim=use_point_dim, sep=sep),
            FilterSmallBBox(size_thr=[0.01, 0.01, 0.01])
        ]

        dataset = ApolloPCDataset(
            dataset_root=dataset_root, dataset_list=[dataset_name], mode="train", 
            transforms=transforms, class_names=cls_names, create_gt_database=True)

        database = defaultdict(list)
        msg = "Begin to generate a database for the apollo dataset."

        for data_idx in logger.range(len(dataset), msg=msg):
            sample = dataset[data_idx]
            image_idx = int(sample.meta.id.split('/')[-1])
            points = sample.data
            bboxes_3d = sample.bboxes_3d
            labels = sample.labels

            num_bboxes = len(bboxes_3d)
            if num_bboxes == 0:
                continue

            masks = get_mask_of_points_in_bboxes3d(
                points, bboxes_3d)  # mask shape: [num_points, num_bboxes]
            for box_idx in range(num_bboxes):
                cls_name = labels[box_idx]
                mask = masks[:, box_idx]
                selected_points = points[mask]
                selected_points[:, 0:3] -= bboxes_3d[box_idx, 0:3]
                num_points = selected_points.shape[0]

                if num_points == 0:
                    warnings.warn("{} frame {}^th box is empty! size is {}, {}, {}".\
                        format(image_idx, box_idx, bboxes_3d[box_idx][3], 
                        bboxes_3d[box_idx][4], bboxes_3d[box_idx][5]))
                    continue
                
                if cls_name.lower() not in class_information:
                    difficulty = 0
                else:
                    if num_points <= class_information[cls_name.lower()]['difficulty_threshold'][0]:
                        difficulty = 2
                    elif num_points <= class_information[cls_name.lower()]['difficulty_threshold'][1]:
                        difficulty = 1
                    else:
                        difficulty = 0

                if not osp.exists(osp.join(save_dir, cls_name)):
                    os.makedirs(osp.join(save_dir, cls_name))
                lidar_file = osp.join(
                    osp.join(save_dir, cls_name), "{}_{}_{}.bin".format(
                        image_idx, cls_name, box_idx))

                with open(lidar_file, "w") as f:
                    selected_points.tofile(f)

                anno_info = {
                    "dataset":
                    dataset_name,
                    "lidar_file":
                    osp.join("apollo_train_gt_database", cls_name,
                            "{}_{}_{}.bin".format(image_idx, cls_name, box_idx)),
                    "cls_name":
                    cls_name,
                    "bbox_3d":
                    bboxes_3d[box_idx, :],
                    "box_idx":
                    box_idx,
                    "data_idx":
                    image_idx,
                    "num_points_in_box":
                    num_points,
                    "lidar_dim":
                    use_point_dim,
                    "difficulty":
                    difficulty
                }
                database[cls_name].append(anno_info)

        db_anno_file = osp.join(osp.join(save_dir, 'anno_info_train.pkl'))
        with open(db_anno_file, 'wb') as f:
            pickle.dump(database, f)
    logger.info("The database generation has been done.")

def generate_nuscenes_gt_database(dataset_root: str,
                                  class_names: List[str] = None,
                                  save_dir: str = None,
                                  max_sweeps: int = 10,
                                  load_point_dim: int = 5,
                                  use_point_dim: int = 4,
                                  use_time_lag: bool = True,
                                  sweep_remove_radius: int = 1):
    if save_dir is None:
        save_dir = dataset_root

    save_dir = osp.join(
        save_dir, "gt_database_train_nsweeps{}_withvelo".format(max_sweeps))

    transforms = [
        LoadPointCloud(
            dim=load_point_dim,
            use_dim=use_point_dim,
            use_time_lag=use_time_lag,
            sweep_remove_radius=sweep_remove_radius)
    ]
    dataset = NuscenesPCDataset(
        dataset_root=dataset_root,
        mode='train',
        transforms=transforms,
        max_sweeps=max_sweeps,
        class_names=class_names)

    for cls_name in dataset.class_names:
        if not osp.exists(osp.join(save_dir, cls_name)):
            os.makedirs(osp.join(save_dir, cls_name))

    database = defaultdict(list)
    msg = "Begin to generate a database for the nuscenes dataset."

    for data_idx in logger.range(len(dataset), msg=msg):
        sample = dataset[data_idx]
        points = sample.data
        bboxes_3d = sample.bboxes_3d
        velocities = sample.bboxes_3d.velocities
        labels = sample.labels

        num_bboxes = len(bboxes_3d)
        if num_bboxes == 0:
            continue
        masks = get_mask_of_points_in_bboxes3d(
            points, bboxes_3d)  # mask shape: [num_points, num_bboxes]
        for box_idx in range(num_bboxes):
            mask = masks[:, box_idx]
            selected_points = points[mask]
            if len(selected_points) == 0:
                continue
            selected_points[:, 0:3] -= bboxes_3d[box_idx, 0:3]

            cls_name = dataset.class_names[labels[box_idx]]
            if not osp.exists(osp.join(save_dir, cls_name)):
                os.makedirs(osp.join(save_dir, cls_name))
            lidar_file = osp.join(
                osp.join(save_dir, cls_name), "{}_{}_{}.bin".format(
                    data_idx, cls_name, box_idx))

            with open(lidar_file, "w") as f:
                selected_points.tofile(f)

            anno_info = {
                "lidar_file":
                osp.join(
                    "gt_database_train_nsweeps{}_withvelo".format(max_sweeps),
                    osp.join(cls_name, "{}_{}_{}.bin".format(
                        data_idx, cls_name, box_idx))),
                "cls_name":
                cls_name,
                "bbox_3d":
                bboxes_3d[box_idx, :],
                "velocity":
                velocities[box_idx, :],
                "box_idx":
                box_idx,
                "data_idx":
                data_idx,
                "num_points_in_box":
                selected_points.shape[0],
                "lidar_dim":
                load_point_dim
            }
            database[cls_name].append(anno_info)

    db_anno_file = osp.join(
        osp.join(save_dir,
                 'anno_info_train_nsweeps{}_withvelo.pkl'.format(max_sweeps)))
    with open(db_anno_file, 'wb') as f:
        pickle.dump(database, f)
    logger.info("The database generation has been done.")
