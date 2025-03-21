# 用于非DLJSFuzzer方法，cell不重复，不添加任何操作
import copy

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class GeneralPaddleNet(nn.Layer):
    def __init__(self,in_channel,final_module,channels):
        super(GeneralPaddleNet, self).__init__()
        self.in_channel = in_channel
        self.final_module = final_module
        self.channels = channels
    def forward(self,x):
        # 各节点的张量。
        tensors = []
        # 判断某张量是否有初始值。
        tensors_isnull = [True] * len(self.channels)
        tensors.append(x)
        tensors_isnull[0] = False
        for i in range(len(self.channels) - 1):
            # 随意赋一个同类型的初始值
            tensors.append(True)
        final_point = 0
        for eachOperation in self.final_module:
            fromIndex = eachOperation.fromIndex
            final_point = eachOperation.toIndex
            input = tensors[fromIndex]
            toIndex = eachOperation.toIndex
            # 本NAS规定所有操作出通道数与入通道数相同。
            operator_in_channel = self.channels[fromIndex]*self.in_channel
            operator = eachOperation.operator

            # if operator != -1 and operator != 0:
            #     operator = -1
            # indentity
            if operator == -1:
                # print("pytorch执行了了操作-1 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    tensors[toIndex] = tensors[fromIndex].clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    temp = paddle.concat([tensors[toIndex], input], 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            # 1 × 1 convolution of C channels
            elif operator == 1:
                # print("pytorch执行了操作1 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # 这是1*1卷积的代码
                    # filter参数顺序:OutChannel、InChannel、H、W
                    filter = paddle.ones([self.in_channel, operator_in_channel, 1, 1])
                    thisresult = F.conv2d(input, weight=filter, stride=[1, 1], padding=0)
                    tensors[toIndex] = thisresult.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    # 这是1*1卷积的代码。
                    filter = paddle.ones([self.in_channel, operator_in_channel, 1, 1])
                    thisresult = F.conv2d(input, weight=filter, stride=[1, 1], padding=0)
                    tensors[toIndex] = paddle.concat([tensors[toIndex], thisresult], 1).clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            # 3 × 3 depthwise convolution
            elif operator == 2:
                # print("pytorch执行了操作2", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # filter参数顺序:OutChannel、InChannel/groups、H、W
                    filter = paddle.ones([operator_in_channel, 1, 3, 3], dtype=paddle.float32)
                    # 注：pytorch中dw卷积加pw卷积是普通卷积操作加groups来调节的。
                    # thisresult = F.conv2d(input=input,weight=filter,stride=1,padding=[1,1],groups=operator_in_channel)
                    pad = paddle.nn.ZeroPad2D(padding=1)
                    input = pad(input)
                    thisresult = F.conv2d(input, weight=filter, stride=1, padding=0,
                                                            groups=operator_in_channel)
                    tensors[toIndex] = thisresult.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    filter = paddle.ones([operator_in_channel, 1, 3, 3], dtype=paddle.float32)
                    # thisresult = F.conv2d(input=input,weight=filter,stride=1,padding=[1,1],groups=operator_in_channel)
                    pad = paddle.nn.ZeroPad2D(padding=(1, 1, 1, 1))
                    input = pad(input)
                    thisresult = F.conv2d(input, weight=filter, stride=1, padding=0,
                                                            groups=operator_in_channel)
                    tensors[toIndex] = paddle.concat([tensors[toIndex], thisresult], 1).clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            elif operator == 3:
                # print("pytorch执行了操作3", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # filter参数顺序:OutChannel、InChannel/groups、H、W
                    depthwise_filter = paddle.ones([operator_in_channel, 1, 3, 3], dtype=paddle.float32)
                    pointwise_filter = paddle.ones([self.in_channel, operator_in_channel, 1, 1], dtype=paddle.float32)
                    pad = paddle.nn.ZeroPad2D(padding=(1, 1, 1, 1))
                    input = pad(input)
                    depthwise_temp = paddle.nn.functional.conv2d(input, weight=depthwise_filter, stride=1,
                                                                padding=0, groups=operator_in_channel).clone().detach()
                    pointwise_temp = paddle.nn.functional.conv2d(depthwise_temp, weight=pointwise_filter, stride=1,
                                                                padding=0).clone().detach()
                    tensors[toIndex] = pointwise_temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    depthwise_filter = paddle.ones([operator_in_channel, 1, 3, 3], dtype=paddle.float32)
                    pointwise_filter = paddle.ones([self.in_channel, operator_in_channel, 1, 1], dtype=paddle.float32)
                    pad = paddle.nn.ZeroPad2D(padding=(1, 1, 1, 1))
                    input = pad(input)
                    depthwise_temp = paddle.nn.functional.conv2d(input, weight=depthwise_filter, stride=1,
                                                                padding=0, groups=operator_in_channel).clone().detach()
                    pointwise_temp = paddle.nn.functional.conv2d(depthwise_temp, weight=pointwise_filter, stride=1,
                                                                padding=0).clone().detach()
                    tensors[toIndex] = paddle.concat([tensors[toIndex], pointwise_temp], 1).clone().detach()
            elif operator == 4:
                # print("pytorch执行了了操作4 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = paddle.nn.MaxPool2D(kernel_size=3, stride=1, padding=1, ceil_mode=False)(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = paddle.nn.MaxPool2D(kernel_size=3, stride=1, padding=1, ceil_mode=False)(input)
                    temp = paddle.concat([tensors[toIndex], result], 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            elif operator == 5:
                # print("pytorch执行了了操作5 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # 注意：一定要有count_include_pad=False,不计算补的0，和tensorflow保持一致。
                    result = paddle.nn.AvgPool2D(kernel_size=3, stride=1, padding=1, ceil_mode=True)(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = paddle.nn.AvgPool2D(kernel_size=3, stride=1, padding=1, ceil_mode=True)(input)
                    temp = paddle.concat([tensors[toIndex], result], 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            # 3 × 3 convolution of C channels
            elif operator == 6:
                # print("pytorch执行了操作6 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # 这是3*3卷积的代码
                    # filter参数顺序:OutChannel、InChannel、H、W
                    filter = paddle.ones([self.in_channel, operator_in_channel, 3, 3])
                    thisresult = F.conv2d(input, weight=filter, stride=[1, 1], padding=1)
                    tensors[toIndex] = thisresult.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    # 这是3*3卷积的代码。
                    filter = paddle.ones([self.in_channel, operator_in_channel, 3, 3])
                    thisresult = F.conv2d(input, weight=filter, stride=[1, 1], padding=1)
                    tensors[toIndex] = paddle.concat([tensors[toIndex], thisresult], 1).clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #3*3 conv2D_transpose of C channels
            #注：参数代表的是正向卷积的过程，但我要做的是反卷积。
            elif operator == 7:
                # print("pytorch执行了操作7 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # 这是3*3卷积的代码
                    # filter参数顺序:OutChannel、InChannel、H、W
                    filter = paddle.ones([operator_in_channel, self.in_channel, 3, 3])
                    thisresult = F.conv2d_transpose(input, weight=filter, stride=[1, 1], padding=1)
                    tensors[toIndex] = thisresult.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    # 这是3*3卷积的代码。
                    filter = paddle.ones([operator_in_channel, self.in_channel, 3, 3])
                    thisresult = F.conv2d_transpose(input, weight=filter, stride=[1, 1], padding=1)
                    tensors[toIndex] = paddle.concat([tensors[toIndex], thisresult], 1).clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #relu
            elif operator == 8:
                # print("pytorch执行了了操作8 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = F.relu(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = F.relu(input)
                    temp = paddle.concat([tensors[toIndex], result], 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #sigmoid
            elif operator == 9:
                # print("pytorch执行了了操作9 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = paddle.nn.functional.sigmoid(input)
                    tensors[toIndex] = result
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = paddle.nn.functional.sigmoid(input)
                    temp = paddle.concat([tensors[toIndex], result], 1)
                    tensors[toIndex] = temp
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #tanh
            elif operator == 10:
                # print("pytorch执行了了操作10 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = paddle.nn.functional.tanh(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = paddle.nn.functional.tanh(input)
                    temp = paddle.concat((tensors[toIndex], result), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #leakyrelu
            elif operator == 11:
                # print("pytorch执行了了操作11 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = paddle.nn.LeakyReLU(negative_slope=0.2)(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = paddle.nn.LeakyReLU(negative_slope=0.2)(input)
                    temp = paddle.concat([tensors[toIndex], result], 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #prelu。
            elif operator == 12:
                # print("pytorch执行了了操作12 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    weight = paddle.ones([operator_in_channel])*0.2
                    result = paddle.nn.functional.prelu(input,weight)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    weight = paddle.ones([operator_in_channel])*0.2
                    result = paddle.nn.functional.prelu(input,weight)
                    temp = paddle.concat([tensors[toIndex], result], 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #ELU
            elif operator == 13:
                # print("pytorch执行了了操作13 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = paddle.nn.ELU()(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = paddle.nn.ELU()(input)
                    temp = paddle.concat([tensors[toIndex], result], 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])


        return tensors[final_point]