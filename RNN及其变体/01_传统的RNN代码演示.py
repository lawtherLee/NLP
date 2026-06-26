"""
案例:
    传统的RNN代码演示.

RNN(Recurrent neural network, 循环神经网络), 主要处理 序列数据 相关问题的.
    分类:
        按照输入和输出分:
            N vs N:
            N vs 1:
            1 vs N:
            N vs M:   seq2seq -> 用的比较多的.
        按照内部结构分:
            传统的RNN:
            LSTM
            Bi-LSTM
            GRU
            Bi-GRU

传统RNN优缺点:
    优点:
        内部结构简单, 资源消耗相对较小.
    缺点:
        处理长序列数据时, 因为反向传播结合梯度连乘, 过大或者过小的w值都会导致 梯度爆炸或者梯度消失.
"""

# 导包
import torch  # 用于计算的
import torch.nn as nn  # 神经网络模块, 里边有各种规则和方法


# todo 1.定义函数, 演示: RNN模型基础版代码.
def dm01():
    # 需求: 创建基础版的RNN模型, 并演示其工作流程.
    # 类比: 玩一个"根据线索猜动物"的游戏.

    # 1. 创建RNN模型 -> 等价于: 设定游戏规则.
    # 参1: 词向量维度(词嵌入维度), 参2: 隐藏层的(输出)维度, 参3: 隐藏层的数量.
    # 参1: 每个线索用5个数字表示(如: 腿的数量, 是否会飞, 是否有毛发, 是否生活在水里, 体型大小)
    # 参2: 大脑一次能记住6条信息(记忆容量)
    # 参3: 只有1层大脑来处理信息.
    rnn = nn.RNN(input_size=5, hidden_size=6, num_layers=1)

    # 2. 准备输入数据 -> 等价于: 生成3个动物的线索, 每个线索用5个数字表示
    # 参1: 句子长度(sequence_length), 参2: 批次的样本数量(batch_size), 参3: 词向量维度(input_size)
    # 参1: 每个只给1个线索.
    # 参2: 同时猜3只动物.
    # 参3: 每个线索用5个数字表示.
    input = torch.randn(1, 3, 5)

    # 3. 初始化隐藏状态 -> 等价于: 创建一个空的记忆(游戏开始前 大脑的空白记忆状态)
    # 参1: num_layer隐藏层层数 * num_directions网络方向, 参2: batch_size(批次的样本数), 参3: hidden_size(隐藏层输出维度)
    # 参1: 只有1层大脑
    # 参2: 同时为3个动物准备记忆
    # 参3: 每个记忆槽能存6个信息.
    h0 = torch.randn(1, 3, 6)

    # 4. 运行模型 -> 等价于: 把线索和初始化记忆输入大脑, 开始思考, 猜动物.
    # 参1: 本次的输入,  参2: 上一时刻的隐藏状态.
    # output(本次的输出): 大脑基于每个线索后的 猜想结果.
    # hn(本次的隐藏状态): 大脑在最后1个线索后的最终记忆状态.
    output, hn = rnn(input, h0)

    # 5. 输出结果
    print(f'output: {output}, {output.shape}')  # shape: (1, 3, 6)
    print(f'hn: {hn}, {hn.shape}')              # shape: (1, 3, 6)
    print(f'rnn模型 -> {rnn}')                   # RNN(5, 6)


# todo 2.定义函数, 演示: RNN模型升级版代码, 修改句子长度, 实际开发中, 1个句子的长度不可能为1的, 效率太慢了.
def dm02():
    # 1. 创建RNN模型
    # 参1: 词向量维度 -> 充当: 输入
    # 参2: 隐藏层的维度 -> 充当: 输出
    # 参3: 隐藏层的数量, 默认是: 1
    rnn = nn.RNN(5, 6, 1)

    # 2. 准备输入数据
    # 参1: sequence_length: 句子的长度(每个句子由多少个词)
    # 参2: batch_size: 批次的样本数量(每批多少个样本)
    # 参3: input_size: 词向量维度(词嵌入维度)
    input = torch.randn(20, 3, 5)

    # 3. 初始化隐藏状态.
    # 参1: num_layer隐藏层层数 * num_directions网络方向,
    # 参2: batch_size(批次的样本数),
    # 参3: hidden_size(隐藏层输出维度)
    h0 = torch.randn(1, 3, 6)

    # 4. 运行模型.
    # output: 本次的输出
    # hn: 本次的隐藏状态
    output, hn = rnn(input, h0)

    # 5. 打印结果.
    print(f'output: {output}, {output.shape}')  # (20, 3, 6)
    print(f'hn: {hn}, {hn.shape}')              # (1, 3, 6)
    print(f'rnn模型 -> {rnn}')                   # rnn(5, 6)


# todo 3.定义函数, 演示: RNN模型升级版代码, 修改隐藏层层数.
def dm03():
    # 1. 创建RNN模型
    # 参1: 词向量维度 -> 充当: 输入
    # 参2: 隐藏层的维度 -> 充当: 输出
    # 参3: 隐藏层的数量, 默认是: 1, 这里我们修改为2
    rnn = nn.RNN(5, 6, 2)

    # 2. 准备输入数据
    # 参1: sequence_length: 句子的长度(每个句子由多少个词)
    # 参2: batch_size: 批次的样本数量(每批多少个样本)
    # 参3: input_size: 词向量维度(词嵌入维度)
    input = torch.randn(20, 3, 5)

    # 3. 初始化隐藏状态.
    # 参1: num_layer隐藏层层数 * num_directions网络方向,
    # 参2: batch_size(批次的样本数),
    # 参3: hidden_size(隐藏层输出维度)
    h0 = torch.randn(2, 3, 6)

    # 4. 运行模型.
    # output: 本次的输出
    # hn: 本次的隐藏状态
    output, hn = rnn(input, h0)

    # 5. 打印结果.
    print(f'output: {output}, {output.shape}')  # (20, 3, 6)
    print(f'hn: {hn}, {hn.shape}')              # (2, 3, 6)
    print(f'rnn模型 -> {rnn}')                   # rnn(5, 6, 2)



# todo 4.测试
if __name__ == '__main__':
    # dm01()
    # dm02()
    dm03()