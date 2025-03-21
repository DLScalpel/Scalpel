# coding=utf-8

from .flatMap import toFlatMap

def calFitness(g):
    #todo
    toFlatMap(g)
    #一定要用到的时候再import，否则会初始化错误的值
    from deploy import deploy
    diff = deploy()
    return diff
