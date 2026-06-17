"""
案例:
    本意是演示下 文本张量的 表示方法, 即:
        one-hot 编码
        Word2Vec
        Word Embedding
    的用法, 但是这部分的面试题有点多, 我们先简单回顾下 和 维度相关的"面试题"

    文本张量表示的作用:
        把文本 表示成 张量矩阵的形式.

结论:
    几个冒号 = 几维
    具体的公式:
        结果张量的维度个位 = 冒号的数量 + 省略的维度个数(省略的:)
"""

import torch

# 模拟真题, 面试的时候, 可能会出的填空题.
torch.manual_seed(30)

# 场景1: 入门版.
a = torch.randn(2, 3, 4)
print(f'a: {a}')                # 2个3行4列的矩阵
print(f'a.shape: {a.shape}')    # 3维, (2, 3, 4)
print('-' * 30)

# 场景2: 升级版.
# print(a[:1])
# print(a[:1].shape)              # 3维, (1, 3, 4)
# print('-' * 30)
#
# # 场景3: 升级版
# print(a[:1, :2])
# print(a[:1, :2].shape)          # 3维, (1, 2, 4)
# print('-' * 30)
#
# # 场景4: 升级版
# print(a[1, :2, 3])
# print(a[1, :2, 3].shape)        # (2)
# print('-' * 30)

# 场景5: 升级版
# print(a[:, 3, :2])              # 越界了
# print(a[:, 3, :2].shape)        # 越界了
# print('-' * 30)

# 场景6: 升级版
print(a[:, 1, :2])
print(a[:, 1, :2].shape)         # 2维, (2, 2)