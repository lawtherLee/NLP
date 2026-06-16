"""
案例:
    代码演示 POS(词性标注)

名词解释:
    NER(命名实体识别, Named Entity Recognition), 就是识别出一段文本中可能存在的 命名实体.
        命名实体:
            人名, 地名, 机构名等专有名词 -> 命名实体
        作用:
            和词汇一样, 也是人类理解文本的基础单元.
            (后续我们NLP的第1个项目, 就是NER, 后续详解, 目前了解概念即可)


    POS(词性标注 Part Of Speech Tagging), 语言中对词的一种分类方法.
        以 语法特征 为主要依据, 兼顾词汇意义对词进行划分 词性, 例如: 名词, 动词, 形容词...

        大白话解释:
            POS(词性标注) = 标注出一段文本中 每个词汇的 词性.
"""

# 导包
import jieba.posseg as pseg

# 1. 定义变量, 记录: 待分词并标注词性的 文本.
content = '我爱北京天安门'

# 2. 使用 pseg#lcut()方法进行 分词 和 词性标注.
# 结果: 每个元素是1个pair对象, 包括: 词语, 词性
result = pseg.lcut(content)
print(f'result: {result}')

# 3. 获取每组数据, 词 和 词性.
for word, flag in result:
    print(f'词语: {word}, 词性: {flag}')
