# coding=utf-8
import copy
import os
import random

from DataStruct.population import Population
from DataStruct.genetypeQueue import GenetypeQueue
from DataStruct.globalConfig import GlobalConfig
from DataStruct.worker import Worker
from DataStruct.controller import Controller
from DataStruct.generalDataGenerator import GeneralDataGenerator
from DataStruct.generalStructureGenerator import GeneralStructureGenerator
from Method.initialize import initialize
from Method.util import getFinalModule_in_str,getChannels_in_str
import csv
import json

from Method import util

# 本方法用于在globalConfig中初始化，包含Scalpel需要的参数和其它方法需要的参数。
def globalInit():
    # step1:配置globalConfig
    print("正在初始化globalConfig")
    out = open(file='./' + 'result.csv' , mode='w', newline='')
    writer = csv.writer(out,delimiter = ",")
    GlobalConfig.N = 0
    GlobalConfig.alreadyMutatetime = 0
    GlobalConfig.flatOperatorMaps = []
    GlobalConfig.resGenetype = []
    GlobalConfig.P = Population()
    GlobalConfig.Q = GenetypeQueue()
    GlobalConfig.final_module = []
    GlobalConfig.channels = []
    GlobalConfig.outFile = out
    GlobalConfig.writer = writer
    writer.writerow(["No","diff",
                     "channels", "modelbody", "fail_time"])
    out.flush()



# 初始化
globalInit()

print("正在初始化种群")
initialize(GlobalConfig.P)
print("种群初始化完成")
print("开始构建controller节点")
controller = Controller()
print("controller节点构建完成")
print("开始构建worker节点")
worker = Worker()
print("worker节点构建完成")

#主流程
t = 0
avg = 0

print("开始进行突变")

while t < GlobalConfig.maxMutateTime:
    # if t >= 9900 and t < 10000:
    #     start_time = datetime.datetime.now()

    controller.excute()

    # if t >= 9900 and t < 10000:
    #     end_time = datetime.datetime.now()
    #     overall += (end_time - start_time).microseconds
    #     print(start_time)
    #     print(end_time)
    #     print(overall)
    try:

        # if t >= GlobalConfig.maxMutateTime - 50:
        #     start_time = datetime.datetime.now()

        diff = worker.excute()
        print("第" + str(t) + "轮已经完成")

        # 写入结果
        GlobalConfig.writer.writerow([str(t),
                                      str(diff),
                                      getChannels_in_str(),
                                      getFinalModule_in_str(),
                                      str(GlobalConfig.fail_time)])

    except Exception as e:
        GlobalConfig.fail_time += 1
        record_path = f"/tmp/pycharm_project_403/Crush_logs/crush_log_{str(GlobalConfig.fail_time)}.json"
        model_info = util.getFinalModule_in_str_formal()
        js_content = {"model_inf": model_info, "error_message": str(e)}
        js_str = json.dumps(js_content, indent=2)

        current_path = os.path.dirname(__file__)
        os.chdir(current_path)
        with open(record_path, mode='w' , encoding="utf-8") as file:
            file.write(js_str)
            file.close()

        print("本轮突变失败！")
        print(e)
    t = t + 1
    GlobalConfig.alreadyMutatetime = t