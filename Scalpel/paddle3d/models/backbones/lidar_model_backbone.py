# input shape=[2,64,496,432]
# output shape=[2,64,248,216]
from paddle3d.apis import manager

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from DataStruct.globalConfig import GlobalConfig
from Method.Models.general_testnet_paddle import GeneralPaddleNet
from DataStruct.edge import edge

from paddleseg.cvlibs import param_init
from paddleseg.models import layers
from paddleseg.utils import utils

__all__ = [
    "LidarModelBackbone"
]

@manager.BACKBONES.add_component
class LidarModel(nn.Layer):
    def __init__(self,in_channels=128,
                 out_channels=[128, 128, 256],
                 layer_nums=[3, 5, 5],
                 downsample_strides=[2, 2, 2]):
        super(LidarModel, self).__init__()
        self.downsample_strides = downsample_strides
        self.paddlebody_1 = GeneralPaddleNet(64,GlobalConfig.final_module, GlobalConfig.channels)
        self.paddlebody_2 = GeneralPaddleNet(64,GlobalConfig.final_module, GlobalConfig.channels)
        self.paddlebody_3 = GeneralPaddleNet(128,GlobalConfig.final_module, GlobalConfig.channels)
    def forward(self, x):

        x1 = self.paddlebody_1(x)
        weight1 = paddle.ones([64,64*GlobalConfig.channels[-1],1,1])
        # weight1 = paddle.ones([64,64,1,1])
        x1 = paddle.nn.functional.conv2d(x1,weight1)
        x1 = x1[:,:,0:248,0:216]

        x2 = self.paddlebody_2(x1)
        weight2 = paddle.ones([128,64*GlobalConfig.channels[-1],1,1])
        x2 = paddle.nn.functional.conv2d(x2,weight2)
        x2 = x2[:,:,0:124,0:108]

        x3 = self.paddlebody_3(x2)
        weight3 = paddle.ones([256,128*GlobalConfig.channels[-1],1,1])
        x3 = paddle.nn.functional.conv2d(x3,weight3)
        x3 = x3[:,:,0:62,0:54]

        return tuple([x1,x2,x3])

@manager.BACKBONES.add_component
def LidarModelBackbone(**kwargs):
    model = LidarModel(
        **kwargs)
    return model