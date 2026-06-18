"""
案例:
    演示 文本张量(文本的词向量表示形式)的实现方式之 word2vec

word2vec介绍:
    概述:
        它是文本张量的一种实现手段, 基于 one-hot做的优化, 主要有 cbow(连续词袋模式), skipgram(跳字模式)
    其中:
        cbow: 基于上下文预测 中间值
        skipgram: 基于中间值 预测 上下文
    无论:
        是上述的那种方式, 最终都是用 更新后的 权重矩阵, 充当: 词向量矩阵.
    即:
        权重矩阵的第1列, 就是: 第1个词的 word2vec 文本张量的表示形式.
        权重矩阵的第2列, 就是: 第2个词的 word2vec 文本张量的表示形式.
        ...

注意:
    facebook开发的 fasttext包 就是一个开源的 词向量和文本分类工具, 我们直接用它来演示 word2vec.
"""

# 导包
import fasttext         # 你需要pip install fasttext先安装一下, 如果不OK, 尝试用 pip install fasttext-wheel

# todo 1.定义函数, 实现: 训练向量模型, 并保存模型.
def dm01_train_save():
    # 1. 直接开始训练, 以 无监督的方式进行.
    # 无监督模式会学习到词语的分布式表示(词向量, word2vec), 默认是: skipgram, 100维
    my_model = fasttext.train_unsupervised('../data/gz03aa')

    # 2. 保存模型为 -> 二进制文件.
    # 后续可以通过 fasttext.load_model() 加载模型.
    my_model.save_model('../model/gz03_fil9.bin')


# todo 2.定义函数, 实现: 加载模型, 并预测 -> 查看单词对应的词向量.
def dm02_get_word_vector():
    # 1. 加载预训练的fastText模型.
    model = fasttext.load_model('./model/gz03_fil9.bin')
    # 2. 获取单个词的 词向量形式(word2vec)
    results = model.get_word_vector('cat')
    print(f'type: {type(results)}')     # numpy数组
    print(f'shape: {results.shape}')    # (100,)
    print(f'results: {results}')        # 具体的值(词向量值)


# todo 3.定义函数, 实现: 加载模型, 并预测 -> 查看单词的相似度(语义)
def dm03_get_similarity():
    # 1. 加载预训练的fastText模型.
    model = fasttext.load_model('./model/gz03_fil9.bin')
    # 2. 查找和 'dog' 语义最相近的一组单词, 默认: 10个.
    results = model.get_nearest_neighbors('dog')
    # 3. 打印结果.
    print(f'results: {results}')


# todo 4.定义函数, 实现: 模型超参数设定.
def dm04_set_hyper_parameter():
    # 1. 直接开始训练, 以 无监督的方式进行.
    # 无监督模式会学习到词语的分布式表示(词向量, word2vec).
    # 默认参数: skipgram, 100维, 5轮, 0.05学习率, 线程数: cpu数量 - 1
    # my_model = fasttext.train_unsupervised('./data/gz03aa')

    # 手动调整参数.
    my_model = fasttext.train_unsupervised(
        input='./data/gz03aa',
        model='cbow',           # 词向量模型: cbow, skipgram
        dim=50,                 # 词向量维度
        epoch=1,                # 训练轮数
        lr=0.01,                # 学习率
        thread=10               # 线程数
    )

    # 2. 保存模型为 -> 二进制文件.
    # 后续可以通过 fasttext.load_model() 加载模型.
    my_model.save_model('./model/gz03_fil9_new.bin')



# todo 5.测试
if __name__ == '__main__':
    dm01_train_save()
    # dm02_get_word_vector()
    # dm03_get_similarity()
    # dm04_set_hyper_parameter()