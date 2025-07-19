# coding=utf-8

import numpy as np
from DataStruct.population import Population
from DataStruct.genetypeQueue import GenetypeQueue


class GlobalConfig:
    # Record of Model Construction Failure
    fail_time = 0
    # Present Size of Corpus
    N = 0
    # Total Layers of Hierarchical Structure
    L = 1
    # Number of Motifs Contained by Each Layer
    operatorNum = np.array([1])
    # Number of Vertexes of the Motifs in Each Layer
    pointNum = [10]
    # DAG Representation
    flatOperatorMaps = []
    # Total Mutation Time for Cell Search
    maxMutateTime = 10000
    # Record of how many rounds has been carried out
    alreadyMutatetime = 0
    # Corpus
    P = Population()
    # Selected Mutation Materials
    Q = GenetypeQueue()
    # Error Modes: max, avg are available, max is recommended
    error_cal_mode = "max"
    # Total Mutation Times Required in Initialization
    initModelNum = 1
    # Operator Sequence of Final Module
    final_module = []
    # Channels of Each Primitive Operator in Final Module
    channels = []
    # Applied Datasets: random, cifar10, mnist, fashion_mnist, imagenet, sinewave, price and predoo
    dataset = 'random'
    # Batch of Random
    batch = 1
    # Initial Channel for Random
    c0 = 3
    # Height of Random 8 de beishu
    h = 48
    # Width of Random 8 de beishu
    w = 48
    # K of Tournament Algorithm
    k = 1
    # Feedback mode: 0. Only primitive operators 1. Only composite operators 2. Both
    mode = 2
    # Result File
    outFile = None
    # Trigger of CSV Writer
    writer = None
    # Primitive Operators
    basicOps = ['identity', 'None', '1*1', 'depthwise_conv2D', 'separable_conv2D', 'max_pooling2D',
                'average_pooling2D', 'conv2D', 'conv2D_transpose', 'ReLU', 'sigmoid', 'tanh', 'leakyReLU',
                'PReLU', 'ELU']
    # The weight of Primitive Operators(Containing None and Identity)
    basicWeights = [1] * len(basicOps)
    # Probability(tendency) of Selecting Primitive Operator
    basicProp = 0.8

    # 车端模型类型，以及每种类型的模型对应的配置文件。
    modeltype_and_configpath = {'MonoDetection_SingleImage': '/tmp/pycharm_project_403/configs/smoke/smoke_hrnet18_no_dcn_kitti.yml',
                                'MultiViewDetection_Image_3DCoord': '/tmp/pycharm_project_403/configs/petr/petr_vovnet_gridmask_p4_800x320.yml',
                                'MultiViewDetection_Image_3DCoord_History': '/tmp/pycharm_project_403/configs/petr/petrv2_vovnet_gridmask_p4_800x320.yml',
                                'LidarDetection': '/tmp/pycharm_project_403/configs/pointpillars/pointpillars_xyres16_kitti_car.yml'
                                }
    # 本次测试输入模型生成的模型类型，在modeltype_and_configpath键值中的一个。
    this_modeltype = 'MonoDetection_SingleImage'
    # 导出的模型路径 模型类型和文件名对应关系:{'MonoDetection_SingleImage': 'smoke','MultiViewDetection_Image_3DCoord': 'petr_inference','MultiViewDetection_Image_3DCoord_History': 'petrv2_inference','LidarDetection': 'pointpillars'}
    exported_model_path = '/tmp/pycharm_project_403/exported_model/smoke.pdmodel'
    # 导出的模型参数路径
    exported_model_weight_path = '/tmp/pycharm_project_403/exported_model/smoke.pdiparams'
    # tested framework: 'autoware','paddlepaddle'
    tested_framework = 'autoware'
