"""
案例: 演示jieba分词.

分词相关介绍:
    概述:
        分词过程 = 找到 分界符 的过程, 找到分界符就可以分词了.
        每个分词结果 = 1个Token
    常见的分词包:
        jieba:      精确模式, 全模式, 搜索引擎模式...
        IK分词器:    ElasticSearch搜索引擎用的多.
        SnowNLP:    基于概率算法的中文自然语言处理工具包.
        pyltp:      哈工大的
        THULAC:     清华的
        ......
    jieba的作用:
        A.支持多种分词模式
            精确模式:       试图将句子 最精确的切开, 适合: 文本分析.
            全模式:        把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能消除歧义
            搜索引擎模式:   在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词。
        B.支持中文繁体分词
        C.支持用户自定义词典

    总结:
        上述的分词模式, 其实就是 分词粒度不同. 例如 "软件工程" -> 软件工程(粗粒度),    软件, 工程 (细粒度)


额外的话(扩展), 为了不和前边的环境冲突, 建议大家新创建1个沙箱来实现:
    Conda命令:
        conda env list                        查看本机安装的所有沙箱
        conda activate 沙箱名                  切换沙箱
        conda create -n 沙箱名 python==3.10    创建新的沙箱, 且指定Python解释器版本为 3.10
        conda remove -n 沙箱名 --all           删除沙箱

    需要你做的:
        1.新建 nlbbase 沙箱
        2.在PyCharm中新建  NLP_Project/day01  项目
        3.配置PyCharm连接 nlpbase沙箱.
        4.在nlpbase沙箱中安装jieba分词器,  演示分词.

"""

import jieba


# todo 1.定义函数, 演示 jieba精确分词模式, 适用于: 文本分析, 自然语言处理(分词标注, 信息提取等...)
def dm01():
    # 2. 使用jieba进行 精确模式分词(默认模式) -> cut_all = False
    # 获取分词结果(生成器对象) -> 好处: 节省内存, 只能遍历一次.
    result1 = jieba.cut(content, cut_all=False)
    print(f'result1: {result1}')        # 结果是: 生成器对象.

    # 3. 从生成器中获取元素.
    # 思路1: next()函数, 逐个获取下个元素.
    print(next(result1))
    print(next(result1))
    print('-' * 30)

    # 思路2: 遍历方式, 从生成器中获取元素.
    for value in result1:
        print(value)
    print('-' * 30)

    # 4. 如果要列表怎么办?
    # 思路1: 直接把上述的 生成器 -> 列表
    list1 = list(result1)
    print(f'list1: {list1}')

    # 思路2: 切词时, 直接返回 list, 相当于: 语法糖.
    result2 = jieba.lcut(content, cut_all=False)
    print(f'result2: {result2}')


# todo 2.定义函数, 演示 jieba全模式分词, 适用于: 关键词提取, 不需要 严格分词准确性 的场景.
def dm02():
    # 2. 使用jieba进行 全模式分词 -> cut_all = True
    # 获取分词结果(生成器对象) -> 好处: 节省内存, 只能遍历一次.
    result1 = jieba.cut(content, cut_all=True)
    print(f'result1: {result1}')        # 结果是: 生成器对象.

    # 3. 从生成器中获取元素.
    # 思路1: next()函数, 逐个获取下个元素.
    print(next(result1))
    print(next(result1))
    print('-' * 30)

    # 思路2: 遍历方式, 从生成器中获取元素.
    for value in result1:
        print(value)
    print('-' * 30)

    # 4. 如果要列表怎么办?
    # 思路1: 直接把上述的 生成器 -> 列表
    list1 = list(result1)
    print(f'list1: {list1}')

    # 思路2: 切词时, 直接返回 list, 相当于: 语法糖.
    result2 = jieba.lcut(content, cut_all=True)
    print(f'result2(全模式): {result2}')

    # 扩展: 打印下 精确模式分词, 用于和上述的 全模式分词, 进行对比.
    result3 = jieba.lcut(content, cut_all=False)
    print(f'result3(精确模式): {result3}')


# todo 3. 定义函数, 演示 jieba搜索引擎模式分词, 适用于: 搜索引擎, 文本匹配.
"""
解释: 搜索引擎分词模式, 在精确模式分词的基础上, 对长词进行再次切分, 提高召回率. 
例如:
    场景1: 用户录入  "程序员"
        精确模式:       只能匹配包含完整 "程序员" 的文档.
        搜索引擎模式:    不仅能匹配 "程序员" 的文档, 还能匹配 "程序", "员"的文档.   提高召回.
    
    场景2: 用户搜索 "黑马程序"
        精确模式:       无法匹配, 分词为("黑马", "程序员")
        搜索引擎模式:    能匹配, 分词为("黑马", "程序", "员")
        
    场景3: 实际应用场景(电商搜索),  商品标题为: "苹果手机保护套", 用户搜索 "苹果套"
        精确模式:       无法匹配, 分词为("苹果", "手机", "保护套")
        搜索引擎模式:    能匹配, 分词为("苹果", "手机", "保护", "套")
        
"""
def dm03():
    # 2. 使用jieba进行 搜索引擎模式分词 -> cut_for_search
    # 获取分词结果(生成器对象) -> 好处: 节省内存, 只能遍历一次.
    result1 = jieba.cut_for_search(content)
    print(f'result1: {result1}')        # 结果是: 生成器对象.

    # 3. 从生成器中获取元素.
    # 思路1: next()函数, 逐个获取下个元素.
    print(next(result1))
    print(next(result1))
    print('-' * 30)

    # 思路2: 遍历方式, 从生成器中获取元素.
    for value in result1:
        print(value)
    print('-' * 30)

    # 4. 如果要列表怎么办?
    # 思路1: 直接把上述的 生成器 -> 列表
    list1 = list(result1)
    print(f'list1: {list1}')

    # 思路2: 切词时, 直接返回 list, 相当于: 语法糖.
    result2 = jieba.lcut_for_search(content)
    print(f'result2(搜索引擎模式): {result2}')

    # 扩展: 打印下 精确模式分词, 用于和上述的 全模式分词, 进行对比.
    result3 = jieba.lcut(content, cut_all=False)
    print(f'result3(精确模式): {result3}')

    # 扩展: 打印下 全模式分词, 用于和上述的 搜索引擎模式分词, 进行对比.
    result4 = jieba.lcut(content, cut_all=True)
    print(f'result4(全模式): {result4}')


# todo 4. 定义函数, 演示 jieba繁体分词, 适用于: 中国香港, 台湾, 澳门...
def dm04():
    # 1. 定义待分词的文本内容.
    content = '煩惱即是菩提，我暫且不提'

    # 2. 切割繁体字.
    result1 = jieba.lcut(content)
    print(f'result1: {result1}')


# todo 5.定义函数, 演示 jieba分词之 自定义词典, 适用于: 稍微生僻点的词或者特殊要求的词组, 一般不会太多.
"""
添加自定义词典后, jieba能够准确识别词典中出现的词汇，提升整体的识别准确率。
词典格式: 每一行分三部分：词语、词频（可省略）、词性（可省略），用空格隔开，顺序不可颠倒。
词典样式如下, 具体词性含义请参照7 jieba词性对照表, 将该词典存为userdict.txt, 方便之后加载使用。
"""
def dm05():
    # 1. 定义待分词的文本内容.
    # 2. 执行未加载自定义词典的分词.
    result1 = jieba.lcut(content)       # 默认: 精确模式分词.
    print(f'result1(精确模式分词): {result1}')

    # 3. 加载用户自定义词典.
    jieba.load_userdict('./data/userdict.txt')
    # 4. 执行加载自定义词典的分词.
    result2 = jieba.lcut(content)
    # 5. 打印结果.
    print(f'result2: {result2}')

# todo 6.测试
if __name__ == '__main__':
    # 1. 定义待分词的文本内容.
    content = '传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能'
    # dm01()
    # dm02()
    dm03()
    # dm04()
    # dm05()