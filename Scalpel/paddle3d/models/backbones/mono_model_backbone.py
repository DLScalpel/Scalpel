# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddle3d.apis import manager

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from Method.Models.general_testnet_paddle import GeneralPaddleNet
from DataStruct.globalConfig import GlobalConfig
from DataStruct.edge import edge

# from paddleseg.cvlibs import manager
from paddleseg.cvlibs import param_init
from paddleseg.models import layers
from paddleseg.utils import utils

__all__ = [
    "MonoModelBackbone"
]

@manager.BACKBONES.add_component
class MonoModel(nn.Layer):
    """
    The HRNet implementation based on PaddlePaddle.

    The original article refers to
    Jingdong Wang, et, al. "HRNet：Deep High-Resolution Representation Learning for Visual Recognition"
    (https://arxiv.org/pdf/1908.07919.pdf).

    Args:
        in_channels (int, optional): The channels of input image. Default: 3.
        pretrained (str, optional): The path of pretrained model.
        stage1_num_modules (int, optional): Number of modules for stage1. Default 1.
        stage1_num_blocks (list, optional): Number of blocks per module for stage1. Default (4).
        stage1_num_channels (list, optional): Number of channels per branch for stage1. Default (64).
        stage2_num_modules (int, optional): Number of modules for stage2. Default 1.
        stage2_num_blocks (list, optional): Number of blocks per module for stage2. Default (4, 4).
        stage2_num_channels (list, optional): Number of channels per branch for stage2. Default (18, 36).
        stage3_num_modules (int, optional): Number of modules for stage3. Default 4.
        stage3_num_blocks (list, optional): Number of blocks per module for stage3. Default (4, 4, 4).
        stage3_num_channels (list, optional): Number of channels per branch for stage3. Default [18, 36, 72).
        stage4_num_modules (int, optional): Number of modules for stage4. Default 3.
        stage4_num_blocks (list, optional): Number of blocks per module for stage4. Default (4, 4, 4, 4).
        stage4_num_channels (list, optional): Number of channels per branch for stage4. Default (18, 36, 72. 144).
        has_se (bool, optional): Whether to use Squeeze-and-Excitation module. Default False.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        use_psa (bool, optional): Usage of the polarized self attention moudle. Default False.
    """

    def __init__(self):
        super(MonoModel, self).__init__()
        self.paddlebody = GeneralPaddleNet(3,GlobalConfig.final_module,GlobalConfig.channels)


        # self.init_weight()

    def forward(self, x):
        x = self.paddlebody(x)
        # 输出维度要求：[8,270,96,320],batch数不变，channel数用1*1卷积统一，hw数用F.interpolate采样解决
        weight = paddle.ones([270,3 * GlobalConfig.channels[-1] , 1 , 1])
        x = paddle.nn.functional.conv2d(x , weight)
        x = F.interpolate(x,[96,320], mode = 'bilinear', align_corners = False)

        return x




@manager.BACKBONES.add_component
def MonoModelBackbone(**kwargs):
    model = MonoModel(
        **kwargs)
    return model

