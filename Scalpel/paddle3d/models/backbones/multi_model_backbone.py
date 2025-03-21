from paddle3d.apis import manager

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from Method.Models.general_testnet_paddle import GeneralPaddleNet
from DataStruct.globalConfig import GlobalConfig

__all__ = [
    "MultiModelBackbone"
]

@manager.BACKBONES.add_component
class MultiModel(nn.Layer):
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
        super(MultiModel, self).__init__()
        self.paddlebody_1 = GeneralPaddleNet(3,GlobalConfig.final_module,GlobalConfig.channels)
        self.paddlebody_2 = GeneralPaddleNet(768,GlobalConfig.final_module,GlobalConfig.channels)

        # self.init_weight()

    def forward(self, x):
        outputs = []
        x1 = self.paddlebody_1(x)
        weight1 = paddle.ones([768,3*GlobalConfig.channels[-1],1,1])
        x1 = paddle.nn.functional.conv2d(x1,weight1)
        x1 = x1[:,:,0:32,0:32]
        outputs.append(x1)

        x2 = self.paddlebody_2(x1)
        weight2 = paddle.ones([1024,768*GlobalConfig.channels[-1],1,1])
        x2 = paddle.nn.functional.conv2d(x2,weight2)
        x2 = x2[:,:,0:32,0:32]
        outputs.append(x2)

        return outputs




@manager.BACKBONES.add_component
def MultiModelBackbone(**kwargs):
    model = MultiModel(
        **kwargs)
    return model