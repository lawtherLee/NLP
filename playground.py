import jieba

content = '传智教育是一家上市公司，旗下有黑马程序员品牌。我在黑马这里学习人工智能'

result1 = jieba.cut(content, cut_all=False)

