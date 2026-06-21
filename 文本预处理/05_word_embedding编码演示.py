"""
案例:
    演示文本张量(文本的 词向量表示形式)实现方式之 word embedding.

大白话解释下 word2vec 和 word embedding的区别:
    word2vec:
        先办证, 后干活. 会预先训练出 词向量, 后续再带入模型做其它操作.
    word embedding:
        边办证, 边干活. 训练过程中, 词向量会自动生成(词嵌入层).
"""

# 导包
import torch  # PyTorch深度学习框架
from tensorflow.keras.preprocessing.text import Tokenizer  # 导入词汇映射器
from torch.utils.tensorboard import SummaryWriter  # 可视化 词向量
import jieba  # 导入分词器
import torch.nn as nn  # 导入神经网络模块


# todo 1. 定义函数, 演示: 使用PyTorch的Embedding(词嵌入层)将文本转成 词向量, 并可视化.
def dm01_embedding_show():
    # 1. 文本分词处理.
    # 1.1 定义待处理的文本.
    sentence1 = (
        "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"
    )
    sentence2 = "我爱自然语言处理"
    # 把上述的两句话, 封装成: 列表.
    sentences = [sentence1, sentence2]

    # 1.2 使用jieba进行分词处理.
    # 1.2.1 定义变量, 记录: 分词后的数据 -> 词语列表.
    word_list = []
    # 1.2.2 遍历上述的 句子列表, 获取到每个句子.
    for sentence in sentences:
        # 1.2.3 把分词结果存储到上述的列表中.
        word_list.append(jieba.lcut(sentence))
    # 1.2.3 打印分词结果.
    # [
    #   ['传智', '教育', '是', '一家', '上市公司', '，', '旗下', '有', '黑马', '程序员', '品牌', '。', '我', '是', '在', '黑马', '这里', '学习', '人工智能'],
    #   ['我', '爱', '自然语言', '处理']
    # ]
    # print(f'分词结果: {word_list}')

    # 2. 构建词汇表 并进行 文本数值化(词向量)
    # 2.1 初始化词汇映射器.
    my_tokenizer = Tokenizer()
    # 2.2 拟合训练数据, 统计词频, 并构建 词汇表.
    my_tokenizer.fit_on_texts(word_list)
    # 2.3 查看 词 和 索引映射字典.
    # 内容为: {'是': 1, '黑马': 2, '我': 3, '传智': 4, '教育': 5, '一家': 6, '上市公司': 7, '，': 8, '旗下': 9, '有':
    #        10, '程序员': 11, '品牌': 12, '。': 13, '在': 14, '这里': 15, '学习': 16, '人工智能': 17, '爱': 18, '自然语言': 19, '处理': 20}
    print(f"词和索引的映射字典: {my_tokenizer.word_index}")

    # 2.4 获取去重后的 所有词汇列表.
    my_token_list = my_tokenizer.word_index.values()
    # 格式为: dict_values([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    # print(my_token_list)

    # 2.4 将分词后的文本 -> 转成数字序列.
    seq2id = my_tokenizer.texts_to_sequences(word_list)
    # 格式为: [[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], [3, 18, 19, 20]]
    print(f"文本转成数字序列: {seq2id}")

    # 3. 创建 词向量嵌入层.  把文本(词对应的编号) 转成 词向量.
    # 3.1 初始化词嵌入层.
    # 参1: 词汇表大小, 即: 唯一的词的个数.
    # 参2: 词向量的维度.
    embed = nn.Embedding(num_embeddings=len(my_token_list), embedding_dim=8)

    # 3.2 查看词嵌入层的 权重参数(随机初始化的词向量).
    print(f"embed: {embed.weight.data}")
    print(f"embed.shape: {embed.weight.data.shape}")

    # 4. 词向量可视化
    # 4.1 创建TensorBoard写入器, 将数据写入到 runs 目录.
    my_summary = SummaryWriter(log_dir="./runs")
    # 4.2 将词向量 和 对应的词语 添加到 TensorBoard中.
    # 参1: 词向量矩阵, 形状是2维的, 分别是: [词汇表大小, 词向量维度], 20个词, 每个词的词向量维度是8维.
    # 参2: 对应的词语列表, 用来标注每个点的.
    my_summary.add_embedding(embed.weight.data, my_token_list)

    # 4.3 关闭写入器.
    my_summary.close()

    # 5. 查看每个单词对应的词向量.
    for idx in range(len(my_tokenizer.word_index)):  # idx: 0 ~ 20 包左不包右, 20个词.
        # 5.1 获取当前单词对应的词向量.
        temp_vector = embed(torch.tensor(idx))
        # print(f'词向量: {temp_vector}')

        # 5.2 获取当前索引对应的单词.
        word = my_tokenizer.index_word[idx + 1]
        print(f"单词: {word}, 词向量: {temp_vector.detach().numpy()}")


if __name__ == "__main__":
    dm01_embedding_show()
