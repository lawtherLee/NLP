"""
案例: 演示n-gram特征

n-gram介绍:
    概述:
        就是连续的n个词/字, 把这些连续片段当做1种特征(小词组特征), 帮我们分析文本规律.
    分类:
        uni-gram(1-gram):  把每个 词/字 拆出来
        bi-gram(2-gram):   找连续2个词的组合
        tri-gram(3-gram):  找连续3个词的组合
    目的:
        让计算机更好的理解 文本规律.
"""

# 1. 定义变量, 记录: n-gram的 n值, 一般是2 或者 3
ngram_range = 2  # 表示n-gram的窗口大小, 此处是2-gram(bi-gram)


# 2. 定义函数, 获取 n-gram特征.
def create_ngram_set(input_list):
    # 1. 通过滑动窗口, 生成n-gram
    # i = 0的时候, 获取的元素: [1, 3, 2, 1, 5, 3]
    # i = 1的时候, 获取的元素: [3, 2, 1, 5, 3]

    # sliced_list: [[1, 3, 2, 1, 5, 3], [3, 2, 1, 5, 3]]
    sliced_lists = [input_list[i:] for i in range(ngram_range)]  # i的范围: 0, 1

    # # 2. 使用zip()函数, 对切片列表数据进行组合
    # ngram_tuples = zip(*sliced_lists)

    # # 3. 转换为集合(自动去重), 返回结果.
    # return set(ngram_tuples)

    # 4. 进阶版, 1行搞定.
    return set(zip(*[input_list[i:] for i in range(ngram_range)]))
    # return ngram_tuples


# 3. 测试代码
if __name__ == "__main__":
    input_list = [1, 3, 2, 1, 5, 3]
    res = create_ngram_set(input_list)
    print(res)
