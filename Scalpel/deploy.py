import datetime
import math
import os

import numpy as np
import yaml

import paddle
from paddle3d.apis.config import Config
from paddle3d.models.base import BaseDetectionModel

from DataStruct.globalConfig import GlobalConfig
from paddle3d.utils.logger import logger

from deploy_util.predictor import Predictor
from deploy_util.lidardetection_infer import lidardetection_infer
from deploy_util.monodetection_singleimage_infer import monodetection_singleimage_infer
from deploy_util.multiviewdetection_image_3dcoord_infer import multiviewdetection_image_3dcoord_infer

def generate_apollo_deploy_file(cfg, save_dir: str):
    yml_file = os.path.join(save_dir, 'apollo_deploy.yaml')
    model = cfg.model

    with open(yml_file, 'w') as file:
        # Save the content one by one to ensure the content order of the output file
        file.write('# base information\n')
        yaml.dump({'name': model.apollo_deploy_name}, file)
        yaml.dump({'date': datetime.date.today()}, file)
        yaml.dump({'task_type': '3d_detection'}, file)
        yaml.dump({'sensor_type': model.sensor}, file)
        yaml.dump({'framework': 'PaddlePaddle'}, file)

        file.write('\n# dataset information\n')
        yaml.dump({
            'dataset': {
                'name': cfg.train_dataset.name,
                'labels': cfg.train_dataset.labels
            }
        }, file)

        file.write('\n# model information\n')
        transforms = cfg.export_config.get('transforms', [])
        save_name = cfg.model.save_name
        model_file = '{}.pdmodel'.format(save_name)
        params_file = '{}.pdiparams'.format(save_name)
        data = {
            'model': {
                'inputs':
                model.inputs,
                'outputs':
                model.outputs,
                'preprocess':
                transforms,
                'model_files':
                [{
                    'name':
                    model_file,
                    'type':
                    'model',
                    'size':
                    os.path.getsize(os.path.join(save_dir, model_file))
                },
                 {
                     'name':
                     params_file,
                     'type':
                     'params',
                     'size':
                     os.path.getsize(os.path.join(save_dir, params_file))
                 }]
            }
        }

        yaml.dump(data, file)


def export(model_config_path):
    cfg = Config(path=model_config_path)
    model = cfg.model
    model.eval()

    # # 导出的时候是否做量化
    # if args.quant_config:
    #     quant_config = get_qat_config(args.quant_config)
    #     cfg.model.build_slim_model(quant_config['quant_config'])

    # # 导出的时候是否加载预训练模型
    # if args.model is not None:
    #     load_pretrained_model(model, args.model)

    model.export('./exported_model')

    # # 导出apollo部署需要的配置
    # if not isinstance(model, BaseDetectionModel):
    #     logger.error('Model {} does not support Apollo yet!'.format(
    #         model.__class__.__name__))
    # else:
    #     generate_apollo_deploy_file(cfg, './exported_model')

# mode为'paddlepaddle'则不使用predictor，直接加载模型做推理；mode为'paddleinference'则使用predictor，初始化engine做推理。
def infer(mode,exported_model_path,exported_model_weight_path):
    res = None
    predictor = Predictor(exported_model_path,exported_model_weight_path,gpu_id=0,use_trt=False,trt_precision=0,trt_use_static=False,trt_static_dir=None,collect_shape_info=False,dynamic_shape_file=None)
    if GlobalConfig.this_modeltype == 'MonoDetection_SingleImage':
        res = monodetection_singleimage_infer(mode,predictor)
    elif GlobalConfig.this_modeltype == 'MultiViewDetection_Image_3DCoord':
        res = multiviewdetection_image_3dcoord_infer(mode,predictor)
    elif GlobalConfig.this_modeltype == 'LidarDetection':
        res = lidardetection_infer(mode,predictor)
    return res

# 调用deploy的时候已经修改完globalConfig中的channel和final_module。因此不用传参，在每个模型定义的文件里自己import globalConfig即可。
def deploy():
    modeltype = GlobalConfig.this_modeltype
    model_config_path = GlobalConfig.modeltype_and_configpath[modeltype]
    export(model_config_path)
    exported_model_path = GlobalConfig.exported_model_path
    exported_model_weight_path = GlobalConfig.exported_model_weight_path
    res1 = infer('paddlepaddle',exported_model_path,exported_model_weight_path)
    res1 = res1.numpy()
    res2 = infer('paddleinference',exported_model_path,exported_model_weight_path)
    res3 = infer('autoware',exported_model_path,exported_model_weight_path)
    if GlobalConfig.tested_framework == 'paddlepaddle':
        diff_tensor = res1 - res2
    elif GlobalConfig.tested_framework == 'autoware':
        diff_tensor = res1 - res3
    max_diff = np.max(np.abs(diff_tensor))
    return max_diff