"""
案例:
    演示文本长度规范案例.

文本长度规范解释:
    概述:
        一般模型的输入需要 等尺寸大小的矩阵, 所以需要对 超长文本做截断, 最不足文本进行补齐 -> 文本长度规范.
    实现方式:
        思路1: 第三方包.
            tensorflow#sequence()
        思路2: 纯Python基础代码实现.
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 导包
from tensorflow.keras.preprocessing import sequence


# 定义变量, 记录: 截断补齐长度参数.
cutlen = 10

# todo 1.定义函数, 对输入的文本张量进行长度规范 -> 第三方包.
def padding(x_train):
    # 使用keras的sequence模块中的pad_sequences方法处理.
    # 参1: 待处理的文本张量.
    # 参2: 最大长度.(max length)
    # 参3: truncating, 截断策略: pre(默认, 从序列前端截取), post(从序列后端截取)
    # 参4: padding, 补齐策略: pre(默认, 从序列前端补齐), post(从序列后端补齐)
    return sequence.pad_sequences(x_train, maxlen=cutlen, truncating='post', padding='post')


# todo 2.定义函数, 对输入的文本张量进行长度规范 -> 纯Python基础代码实现.
def padding_custom(data):
    # 1. 定义变量, 记录: 初始化列表.
    list1 = []
    # 2. 遍历每条序列.
    for value in data:
        # 3. 处理超长文本, 截断维度, 保留前cutlen个元素(之类: 默认是10)
        if len(value) > cutlen:
            # 长度超长, 就截断.
            list1.append(value[:cutlen])
        # 4. 走这里, 处理短序列.
        else:
            # 计算需要补齐0的量
            padding_len = cutlen - len(value)
            # 创建补齐列表.
            list1.append(value + [0] * padding_len)

    # 6. 返回处理后的序列列表.
    return list1

# todo 3.测试.
if __name__ == '__main__':
    # 假定x_train里面有两条文本, 一条长度大于10, 一天小于10
    x_train = [[1, 23, 5, 32, 55, 63, 2, 21, 78, 32, 23, 1],
               [2, 32, 1, 23, 1]]

    # 思路1: 第三方包
    # res = padding(x_train)

    # 思路2: 纯Python基础代码实现
    res = padding_custom(x_train)
    print(res)


    import torch.nn as nn
    nn.RNN()

