"""
案例:
    演示 文本分析的常见操作.

文本分析作用:
    帮助我们理解数据语料, 快速检查出语料中可能存在的问题,
    例如:
        数据质量类: 错别字, 语法错误, 重复内容, 缺失值, 噪声....
        分布不均衡问题: 标签分布不均, 句子长度不同...
        ...
文本分析方式:
    1. 标签的数量分布
    2. 句子长度分布
    3. 词频统计和关键字词云
"""
import jieba
# 导包
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt     # 推荐 pip install matplotlib==3.9
from itertools import chain         # 迭代器工具
import jieba.posseg as pseg         # 词性标注(名词, 动词, 形容词..._
from wordcloud import WordCloud     # 词云

plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']   Mac本用这个字体
plt.rcParams['axes.unicode_minus'] = False

# todo 1.定义函数, 实现训练集和测试集的 标签分布的 可视化统计
def dm01_label_sns_countplot():
    # 1. 设置538风格 -> 一种具有现代感的可视化风格(不做也行)
    # plt.style.use('fivethirtyeight')

    # 2. 读取训练集 和 测试集.
    # 参1: 文件路径, 参2: 列分隔符(csv文件用, tsv文件用\t)
    train_data = pd.read_csv('./data/train.tsv', sep='\t')
    dev_data = pd.read_csv('./data/dev.tsv', sep='\t')
    # print(f'train_data.head: {train_data.head()}')
    # print(f'dev_data.head: {dev_data.head()}')

    # 3. 统计训练集标签的 0(负) 和 1(正) 的分组数量并可视化, 采用: 计数柱状图(countplot)
    # 参1: x轴标签, 参2: 数据集, 参3: 用于分组的分类变量, 参4: 是否显示图例(默认显示)
    sns.countplot(x='label', data=train_data, hue='label', legend=False)
    plt.title('train_label')
    plt.show()

    # 4. 统计测试集标签的 0(负) 和 1(正) 的分组数量并可视化, 采用: 计数柱状图(countplot)
    sns.countplot(x='label', data=dev_data, hue='label', legend=False)
    plt.title('dev_label')
    plt.show()


# todo 2.定义函数, 实现训练集和测试集的 句子长度分布的 可视化统计
def dm02_len_sns_countplot_distplot():
    # 1. 读取训练集 和 测试集.
    # 参1: 文件路径, 参2: 列分隔符(csv文件用, tsv文件用\t)
    train_data = pd.read_csv('./data/train.tsv', sep='\t')
    dev_data = pd.read_csv('./data/dev.tsv', sep='\t')

    # 2. 计算训练集的 (每个句子的)长度.
    train_data['sentence_length'] = list(map(lambda x: len(x), train_data['sentence']))
    # 下述代码效果同上, 两种思路均可
    # train_data['sentence_length2'] = train_data['sentence'].apply(lambda x: len(x))
    # print(train_data.iloc[:, 2:])

    # 3. 绘制训练集的 句子长度分布.
    # 图1: 计数柱状图
    sns.countplot(x='sentence_length', data=train_data)
    plt.title('训练集句子长度分布_计数柱状图')
    plt.xticks([])      # 隐藏x轴刻度值
    plt.show()

    # 图2: 密度曲线图
    # distplot() 函数已过时, 可以用, 会报警告, 但是不会报错.
    # sns.distplot(x=train_data['sentence_length'], kde=True)           # seaborn: 3.9.1及其以下写法.
    # sns.distplot(x='sentence_length', data=train_data, kde=True)      # matplotlib: 3.10及其以上写法.

    # histplot(): 直方图
    # sns.histplot(x='sentence_length', data=train_data, kde=True)
    sns.histplot(x=train_data['sentence_length'], kde=True)     # 效果同上.
    plt.title('训练集句子长度分布_密度曲线图')
    plt.show()

    # 4. 绘制测试集的 句子长度分布.
    # 4.1 计算测试集的 (每个句子的)长度.
    dev_data['sentence_length'] = list(map(lambda x: len(x), dev_data['sentence']))

    # 4.2 绘制测试集的 句子长度分布 -> 计数柱状图, 密度曲线图
    # 图1: 计数柱状图
    sns.countplot(x='sentence_length', data=dev_data)
    plt.title('测试集句子长度分布_计数柱状图')
    plt.xticks([])  # 隐藏x轴刻度值
    plt.show()

    # 图2: 密度曲线图
    sns.histplot(x=dev_data['sentence_length'], kde=True)  # 效果同上.
    plt.title('测试集句子长度分布_密度曲线图')
    plt.show()


# todo 3.定义函数, 实现训练集和测试集的 正负样本长度散点分布
def dm03_sns_stripplot():
    # 1. 读取训练集 和 测试集.
    train_data = pd.read_csv('./data/train.tsv', sep='\t')
    dev_data = pd.read_csv('./data/dev.tsv', sep='\t')

    # 2. 获取数据长度列
    # 训练集
    train_data['sentence_length'] = list(map(lambda x: len(x), train_data['sentence']))
    # train_data['sentence_length'] = train_data['sentence'].apply(lambda x: len(x))    # 效果同上.

    # 测试集
    dev_data['sentence_length'] = list(map(lambda x: len(x), dev_data['sentence']))

    # 3. 统计正负样本长度散点分布:
    # 训练集
    sns.stripplot(x='label', y='sentence_length', data=train_data)
    plt.title('训练集正负样本长度散点分布')
    plt.show()

    # 测试集
    sns.stripplot(x='label', y='sentence_length', data=dev_data)
    plt.title('测试集正负样本长度散点分布')
    plt.show()


# todo 4.定义函数, 实现训练集和测试集的 词频统计(即: 获取不同词汇总数统计)
def dm04_word_count_wordcloud():
    # 1. 读取训练集 和 测试集.
    train_data = pd.read_csv('./data/train.tsv', sep='\t')
    dev_data = pd.read_csv('./data/dev.tsv', sep='\t')

    # 2. 统计训练集的词汇总数.
    train_vocab = set(chain(*map(lambda x: jieba.lcut(x), train_data['sentence'])))

    # 3. 统计测试集的词汇总数.
    dev_vocab = set(chain(*map(lambda x: jieba.lcut(x), dev_data['sentence'])))

    # 4. 打印结果.
    print(f'训练集不同词汇总数: {len(train_vocab)}')
    print(f'测试集不同词汇总数: {len(dev_vocab)}')


# todo 5.定义函数, 实现训练集和测试集的 高频形容词词云
# todo 5.1 定义函数, 获取文本中的形容词列表.
def get_a_list(text):       # 形容词用a表示
    # 1. 定义空列表, 用于存储: 文本中的形容词.
    a_list = []

    # 2. 使用jieba的词性标注功能, 切分文本, 遍历整个切分动作.
    for value in pseg.lcut(text):
        # print(f'value: {value}')        # 词 词性
        # print(f'value: {value.word}')   # 词
        # print(f'value: {value.flag}')   # 词性
        # 3. 判断词性, 是否是形容词, 如果是形容词, 则添加到列表中.
        if value.flag == 'a':
            a_list.append(value.word)
    # 4. 返回结果
    return a_list

# todo 5.2 定义函数, 根据词云列表产生词云
def get_word_cloud(keywords_list):
    # 1. 实例化词云生成器, 设置字体路径, 最大显示数, 背景色...
    # 参1: 字体路径, 参2: 最大显示数, 参3: 背景色
    wordcloud = WordCloud(font_path='./data/SimHei.ttf', max_words=100, background_color='white')
    # 2. 将关键词列表 -> 转换为 空格分割的字符串, 适配词云输入格式.
    keywords_str = ' '.join(keywords_list)
    # 3. 根据关键字字符串, 生成词云.
    wordcloud.generate(keywords_str)
    # 4. 配置并显示词云图像.
    # 4.1 创建新的绘图窗口
    plt.figure()
    # 4.2 生成图云(绘制图像)
    # 参1: 生成词云图像的数据, 参2: 设置图像插值方法为 -> 双线性插值
    plt.imshow(wordcloud, interpolation='bilinear')
    # 4.3 隐藏坐标轴
    plt.axis('off')
    # 4.4 显示图像
    plt.show()

# todo 5.3 定义函数, 实现训练集和测试集的 高频形容词词云
def dm05_word_cloud():
    # 场景1: 处理训练集.
    # 1. 读取训练集
    train_data = pd.read_csv('./data/train.tsv', sep='\t')

    # 2. 处理 训练集 正样本(label=1) -> 生成词云.
    # 2.1 筛选label=1的样本, 并提取句子(sentence)列
    p_train_data = train_data[train_data['label'] == 1]['sentence']
    # print(p_train_data)
    # 2.2 对每个正样本句子, 提取形容词, 并用chain合并所有的形容词. (chain: 链式操作)
    p_a_train_vocab = chain(*map(lambda x: get_a_list(x), p_train_data))
    # 2.3 调用词云函数, 根据形容词列表, 生成词云.
    get_word_cloud(p_a_train_vocab)

    # 3. 分隔符.
    print('-' * 60)

    # 4. 处理 训练集 负样本(label=0) -> 生成词云.
    # 4.1 筛选label=1的样本, 并提取句子(sentence)列
    p_train_data = train_data[train_data['label'] == 0]['sentence']
    # print(p_train_data)
    # 4.2 对每个正样本句子, 提取形容词, 并用chain合并所有的形容词. (chain: 链式操作)
    p_a_train_vocab = chain(*map(lambda x: get_a_list(x), p_train_data))
    # 4.3 调用词云函数, 根据形容词列表, 生成词云.
    get_word_cloud(p_a_train_vocab)

if __name__ == '__main__':
    # 1. 调用函数, 演示: 训练集和测试集的 标签分布的 可视化统计
    # dm01_label_sns_countplot()

    # 2. 调用函数, 演示: 训练集和测试集的 句子长度分布的 可视化统计
    # dm02_len_sns_countplot_distplot()

    # 3. 调用函数, 演示: 训练集和测试集的 正负样本长度散点分布
    # dm03_sns_stripplot()

    # 4. 调用函数, 演示: 训练集和测试集的 词频统计
    # dm04_word_count_wordcloud()

    # 5. 调用函数, 演示: 训练集和测试集的 高频形容词词云
    # 5.1 测试: 获取文本中的形容词列表.
    # text = '今天天气不错, 气温正好'
    # print(get_a_list(text))     # ['不错', '正好']

    # 5.2 测试: 根据词云列表产生词云
    # keywords_list = ['不错', '正好', '天气', '气温']    # 模拟的词云列表, 啥词都有.
    # get_word_cloud(keywords_list)

    # 5.3 调用函数, 实现训练集和测试集的 高频形容词词云
    dm05_word_cloud()