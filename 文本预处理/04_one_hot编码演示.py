"""
案例:
    演示 one-hot编码, 分别演示 复杂版 和 简单版.

文本张量相关介绍:
    概述:
        对文本进行切词, 把每个词转成对应的词向量, 即: 用词向量的形式来描述文本 -> 文本张量.
    作用:
        模型无法直接解析文本, 可以转成 文本张量(词向量), 作为模型的输入来进行各种处理.
    (实现)方式:
        one-hot编码:
            独热编码, 01编码, 有这个词就用1表示, 没有这个词就用0表示, 列表长度 = 文本切词去重后总长.
        word2vec:
            CBOW       连续词袋模式
            Skipgram   跳字模式


one-hot编码:
    优点:
        操作简单, 容易理解.
    缺点:
        完全割裂了词与词的关系, 且在大语料数据集的情况下, 每个向量的长度过程, 占用大量内存.   稀疏性.

    解决方案:
        采用 稠密向量表示法: word2vec, word embedding
"""

# 导包   你需要 pip install tensorflow
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# import jieba  # jieba分词
from tensorflow.keras.preprocessing.text import Tokenizer  # 导入keras中的词汇映射器Tokenizer
import joblib  # 导入用于对象保存与加载的joblib


# todo 1.定义函数, 演示: 获取 one-hot编码
def dm_onehot_gen():
    # 1. 准备语料 vocabs
    vocabs = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "光良", "鹿晗"}
    # 2. 实例化 词汇映射器Tokenizer
    my_tokenizer = Tokenizer()
    # 3. 通过 词汇映射器, 在语料库上进行训练.
    my_tokenizer.fit_on_texts(vocabs)
    # 4. 打印word_index, 键: 歌手名, 值: 序号
    # 格式: {'李宗盛': 1, '光良': 2, '周杰伦': 3, '鹿晗': 4, '陈奕迅': 5, '王力宏': 6}
    print(my_tokenizer.word_index)
    # 5. 对每个词(歌手名)进行 one-hot编码
    for vocab in vocabs:
        # 5.1 先创建长度 = 语料库长度的列表, 列表内元素都为0
        zero_list = [0] * len(vocabs)
        # print(f'zero_list: {zero_list}')

        # 5.2 获取当前词汇在word_index中的索引, 因为索引从1开始, 所以索引-1
        idx = my_tokenizer.word_index[vocab] - 1
        # 5.3 修改对应位置的元素为1, 完成 one-hot编码
        zero_list[idx] = 1
        # 5.4 打印结果.
        print(f'当前{vocab}的one-hot编码为: {zero_list}')
    # 6. 保存 词汇映射器
    joblib.dump(my_tokenizer, '../model/my_tokenizer')
    print('模型保存成功!')


# todo 2.定义函数, 演示: 使用 one-hot编码
def use_one_hot():
    # 1. 加载训练好的 词汇映射器
    my_tokenizer = joblib.load('../model/my_tokenizer')
    # 2. 打印加载的 词汇映射器的 word_index, 查看 词汇和索引的对应关系.
    print(f'my_tokenizer.word_index: {my_tokenizer.word_index}')

    # 3. 对指定词汇进行 one-hot编码操作.
    token = '陈奕迅'
    zero_list = [0] * len(my_tokenizer.word_index)
    # 4. 获取指定词汇在word_index中的索引, 因为索引从1开始, 所以索引-1
    idx = my_tokenizer.word_index[token] - 1
    zero_list[idx] = 1
    print(f'当前{token}的one-hot编码为: {zero_list}')


# todo 3.扩展_简单版方式, 实现: 获取 one-hot编码.
def dm03():
    # 1. 准备语料 vocabs
    vocabs = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "光良", "鹿晗"}
    # 2. 构建词汇到索引的映射字典.
    word2index = {value:i for i, value in enumerate(vocabs)}
    print(f'word2index: {word2index}')  # {'鹿晗': 0, '王力宏': 1, '李宗盛': 2, '陈奕迅': 3,...}

    # 3. 对每个词生成one-hot编码.
    for vocab in vocabs:
        # 3.1 初始化全0列表, 长度 = 语料库长度
        zero_list = [0] * len(vocabs)
        # 3.2 获取当前词汇在word2index中的索引, 因为索引从0开始, 所以索引
        idx = word2index[vocab]
        # 3.3 修改对应索引位置的元素值为1, 完成 one-hot编码
        zero_list[idx] = 1
        # 3.4 打印结果.
        print(f'当前 {vocab} 的one-hot编码为: {zero_list}')




# todo 4.测试
if __name__ == '__main__':
    # dm_onehot_gen()       # 获取 one-hot编码
    # use_one_hot()         # 使用 one-hot编码
    dm03()                  # 简易版 获取one-hot编码