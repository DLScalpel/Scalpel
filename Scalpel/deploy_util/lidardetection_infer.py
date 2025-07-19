import numba
import numpy as np
import paddle
from paddle.inference import Config, create_predictor
from deploy_util.predictor import infer_with_onnxruntime_trt_lidar


def read_point(file_path, num_point_dim):
    points = np.fromfile(file_path, np.float32).reshape(-1, num_point_dim)
    points = points[:, :4]
    return points


@numba.jit(nopython=True)
def _points_to_voxel(points, voxel_size, point_cloud_range, grid_size, voxels,
                     coords, num_points_per_voxel, grid_idx_to_voxel_idx,
                     max_points_in_voxel, max_voxel_num):
    num_voxels = 0
    num_points = points.shape[0]
    # x, y, z
    coord = np.zeros(shape=(3, ), dtype=np.int32)

    for point_idx in range(num_points):
        outside = False
        for i in range(3):
            coord[i] = np.floor(
                (points[point_idx, i] - point_cloud_range[i]) / voxel_size[i])
            if coord[i] < 0 or coord[i] >= grid_size[i]:
                outside = True
                break
        if outside:
            continue
        voxel_idx = grid_idx_to_voxel_idx[coord[2], coord[1], coord[0]]
        if voxel_idx == -1:
            voxel_idx = num_voxels
            if num_voxels >= max_voxel_num:
                continue
            num_voxels += 1
            grid_idx_to_voxel_idx[coord[2], coord[1], coord[0]] = voxel_idx
            coords[voxel_idx, 0:3] = coord[::-1]
        curr_num_point = num_points_per_voxel[voxel_idx]
        if curr_num_point < max_points_in_voxel:
            voxels[voxel_idx, curr_num_point] = points[point_idx]
            num_points_per_voxel[voxel_idx] = curr_num_point + 1

    return num_voxels


def hardvoxelize(points, point_cloud_range, voxel_size, max_points_in_voxel,
                 max_voxel_num):
    num_points, num_point_dim = points.shape[0:2]
    point_cloud_range = np.array(point_cloud_range)
    voxel_size = np.array(voxel_size)
    voxels = np.zeros((max_voxel_num, max_points_in_voxel, num_point_dim),
                      dtype=points.dtype)
    coords = np.zeros((max_voxel_num, 3), dtype=np.int32)
    num_points_per_voxel = np.zeros((max_voxel_num, ), dtype=np.int32)
    grid_size = np.round((point_cloud_range[3:6] - point_cloud_range[0:3]) /
                         voxel_size).astype('int32')

    grid_size_x, grid_size_y, grid_size_z = grid_size

    grid_idx_to_voxel_idx = np.full((grid_size_z, grid_size_y, grid_size_x),
                                    -1,
                                    dtype=np.int32)

    num_voxels = _points_to_voxel(points, voxel_size, point_cloud_range,
                                  grid_size, voxels, coords,
                                  num_points_per_voxel, grid_idx_to_voxel_idx,
                                  max_points_in_voxel, max_voxel_num)

    voxels = voxels[:num_voxels]
    coords = coords[:num_voxels]
    num_points_per_voxel = num_points_per_voxel[:num_voxels]

    return voxels, coords, num_points_per_voxel

def preprocess(file_path, num_point_dim, point_cloud_range, voxel_size,
               max_points_in_voxel, max_voxel_num):
    points = read_point(file_path, num_point_dim)
    voxels, coords, num_points_per_voxel = hardvoxelize(
        points, point_cloud_range, voxel_size, max_points_in_voxel,
        max_voxel_num)

    return voxels, coords, num_points_per_voxel

def parse_result(box3d_lidar, label_preds, scores):
    num_bbox3d, bbox3d_dims = box3d_lidar.shape
    for box_idx in range(num_bbox3d):
        # filter fake results: score = -1
        if scores[box_idx] < 0:
            continue
        if bbox3d_dims == 7:
            print(
                "Score: {} Label: {} Box(x_c, y_c, z_c, w, l, h, -rot): {} {} {} {} {} {} {}"
                .format(scores[box_idx], label_preds[box_idx],
                        box3d_lidar[box_idx, 0], box3d_lidar[box_idx, 1],
                        box3d_lidar[box_idx, 2], box3d_lidar[box_idx, 3],
                        box3d_lidar[box_idx, 4], box3d_lidar[box_idx, 5],
                        box3d_lidar[box_idx, 6]))

def run(predictor, voxels, coords, num_points_per_voxel):
    paddle_predictor = predictor.get_predictor()
    input_names = paddle_predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = paddle_predictor.get_input_handle(name)
        if name == "voxels":
            input_tensor.reshape(voxels.shape)
            input_tensor.copy_from_cpu(voxels.copy())
        elif name == "coords":
            input_tensor.reshape(coords.shape)
            input_tensor.copy_from_cpu(coords.copy())
        elif name == "num_points_per_voxel":
            input_tensor.reshape(num_points_per_voxel.shape)
            input_tensor.copy_from_cpu(num_points_per_voxel.copy())
    # do the inference
    paddle_predictor.run()

    # get out data from output tensor
    output_names = paddle_predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = paddle_predictor.get_output_handle(name)
        if i == 0:
            box3d_lidar = output_tensor.copy_to_cpu()
        elif i == 1:
            label_preds = output_tensor.copy_to_cpu()
        elif i == 2:
            scores = output_tensor.copy_to_cpu()
    return box3d_lidar, label_preds, scores
def lidardetection_infer(mode,predictor):
    if mode == 'paddlepaddle':
        # 把字符串里的.pdmodel去掉
        model = paddle.jit.load(predictor.model_file[:-8])

        lidar_file = '/home/zou/桌面/KITTI/training/velodyne/000000.bin'
        num_point_dim = 4
        point_cloud_range = [0.0, -39.68, -3.0, 69.12, 39.68, 1.0]
        voxel_size = [0.16, 0.16, 4.0]
        max_points_in_voxel = 32
        max_voxel_num = 40000

        voxels, coords, num_points_per_voxel = preprocess(
            lidar_file, num_point_dim, point_cloud_range,
            voxel_size, max_points_in_voxel, max_voxel_num)
        box3d_lidar, label_preds, scores = model(coords,num_points_per_voxel,voxels)
        return box3d_lidar
    elif mode == 'paddleinference':

        lidar_file = '/KITTI/training/velodyne/000000.bin'
        num_point_dim = 4
        point_cloud_range = [0.0, -39.68, -3.0, 69.12, 39.68, 1.0]
        voxel_size = [0.16, 0.16, 4.0]
        max_points_in_voxel = 32
        max_voxel_num = 40000

        voxels, coords, num_points_per_voxel = preprocess(
            lidar_file, num_point_dim, point_cloud_range,
            voxel_size, max_points_in_voxel, max_voxel_num)

        box3d_lidar, label_preds, scores = run(predictor, voxels, coords,num_points_per_voxel)
        return box3d_lidar
    elif mode == 'autoware':
        lidar_file = '/KITTI/training/velodyne/000000.bin'
        num_point_dim = 4
        point_cloud_range = [0.0, -39.68, -3.0, 69.12, 39.68, 1.0]
        voxel_size = [0.16, 0.16, 4.0]
        max_points_in_voxel = 32
        max_voxel_num = 40000

        voxels, coords, num_points_per_voxel = preprocess(
            lidar_file, num_point_dim, point_cloud_range,
            voxel_size, max_points_in_voxel, max_voxel_num)
        result = infer_with_onnxruntime_trt_lidar('/tmp/pycharm_project_403/exported_model/smoke.onnx',voxels, coords, num_points_per_voxel)
        return result
