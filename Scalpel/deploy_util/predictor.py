import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import inference

class Predictor():
    model_file = None
    params_file = None
    predictor = None
    def __init__(self, model_file,
                   params_file,
                   gpu_id=0,
                   use_trt=True,
                   trt_precision=0,
                   trt_use_static=False,
                   trt_static_dir=None,
                   collect_shape_info=False,
                   dynamic_shape_file=None):
        self.model_file = model_file
        self.params_file = params_file
        config = inference.Config(model_file, params_file)
        # config.enable_use_gpu(1000, gpu_id)

        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()

        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # create predictor
        if use_trt:
            precision_mode = paddle.inference.PrecisionType.Float32
            if trt_precision == 1:
                precision_mode = paddle.inference.PrecisionType.Half
            config.enable_tensorrt_engine(
                workspace_size=1<<20,
                max_batch_size=1,
                min_subgraph_size=30,
                precision_mode=precision_mode,
                use_static=trt_use_static,
                use_calib_mode=False)
            if collect_shape_info:
                config.collect_shape_range_info(dynamic_shape_file)
            else:
                config.enable_tuned_tensorrt_dynamic_shape(dynamic_shape_file, True)
            if trt_use_static:
                config.set_optim_cache_dir(trt_static_dir)
        self.predictor = inference.create_predictor(config)
    def get_predictor(self):
        return self.predictor


import onnxruntime as ort
def infer_with_onnxruntime_trt_image(onnx_path, img, K, ratio):
    so = ort.SessionOptions()

    # 配置 TensorRT 后端
    trt_options = {
        'device_id': 0,
        'trt_max_workspace_size': 1 << 30,  # 1GB
        'trt_fp16_enable': False,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': './trt_cache'
    }

    # 创建会话
    session = ort.InferenceSession(
        onnx_path,
        providers=['TensorrtExecutionProvider'],
        provider_options=[trt_options],
        sess_options=so
    )

    # 获取输入名称
    input_names = [input.name for input in session.get_inputs()]

    # 准备输入数据
    inputs = {
        input_names[0]: ratio,
        input_names[1]: img,
        input_names[2]: K
    }

    # 执行推理
    outputs = session.run(None, inputs)

    return outputs[0]

##########################################################################
def infer_with_onnxruntime_trt_lidar(onnx_path, voxels, coords, num_points_per_voxel):
    so = ort.SessionOptions()

    # 配置 TensorRT 后端
    trt_options = {
        'device_id': 0,
        'trt_max_workspace_size': 1 << 30,  # 1GB
        'trt_fp16_enable': False,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': './trt_cache'
    }

    # 创建会话
    session = ort.InferenceSession(
        onnx_path,
        providers=['TensorrtExecutionProvider'],
        provider_options=[trt_options],
        sess_options=so
    )

    # 获取输入名称
    input_names = [input.name for input in session.get_inputs()]

    # 准备输入数据
    inputs = {
        input_names[0]: num_points_per_voxel,
        input_names[1]: coords,
        input_names[2]: voxels
    }

    # 执行推理
    outputs = session.run(None, inputs)

    return outputs[0]

###########################################################################
def infer_with_onnxruntime_trt_multi(onnx_path,image,img2lidars):
    so = ort.SessionOptions()

    # 配置 TensorRT 后端
    trt_options = {
        'device_id': 0,
        'trt_max_workspace_size': 1 << 30,  # 1GB
        'trt_fp16_enable': False,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': './trt_cache'
    }

    # 创建会话
    session = ort.InferenceSession(
        onnx_path,
        providers=['TensorrtExecutionProvider'],
        provider_options=[trt_options],
        sess_options=so
    )

    # 获取输入名称
    input_names = [input.name for input in session.get_inputs()]

    # 准备输入数据
    inputs = {
        input_names[0]: img2lidars,
        input_names[1]: image
    }

    # 执行推理
    outputs = session.run(None, inputs)

    return outputs[0]