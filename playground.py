import torch  # PyTorch深度学习框架
from tensorflow.keras.preprocessing.text import Tokenizer  # 导入词汇映射器
from torch.utils.tensorboard import SummaryWriter  # 可视化 词向量
import jieba  # 导入分词器
import torch.nn as nn  # 导入神经网络模块

sentence1 = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"
sentence2 = "我爱自然语言处理"

sentence = [sentence1, sentence2]
# print(sentence)
word_list = []

for sentence in sentence:
    word_list.append(list(jieba.cut(sentence)))

# print(word_list)

my_tokenizer = Tokenizer()
my_tokenizer.fit_on_texts(word_list)

print(my_tokenizer.word_index)

my_token_list = my_tokenizer.word_index.values()
print(my_token_list)

seq2id = my_tokenizer.texts_to_sequences(word_list)

embed = nn.Embedding(len(my_token_list), 8)

my_summary = SummaryWriter(log_dir="./runs")
my_summary.add_embedding(embed.weight.data, my_token_list)

for idx in range(len(my_tokenizer.word_index)):
    temp_vec = embed(torch.tensor(idx))
    word = my_tokenizer.index_word[idx + 1]
    print(f"单词: {word}, 词向量: {temp_vec.detach().numpy()}")
