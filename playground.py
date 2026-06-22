import jieba
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt  # 推荐 pip install matplotlib==3.9
from itertools import chain  # 迭代器工具
import jieba.posseg as pseg  # 词性标注(名词, 动词, 形容词..._
from pandas.io.clipboard import paste
from wordcloud import WordCloud  # 词云

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def dm01_label_sns_counterpoint():
    plt.style.use("fivethirtyeight")

    train_data = pd.read_csv("./data/train.tsv", sep="\t")
    dev_data = pd.read_csv("./data/dev.tsv", sep="\t")
    # print(f"train_data.head: {train_data.head()}")
    # print(f"dev_data.head: {dev_data.head()}")
    sns.countplot(x="label", data=train_data, hue="label", legend=False)
    plt.title("train_label")
    plt.show()


def dm02_len_sns_counterpoint_disappoint():
    train_data = pd.read_csv("./data/train.tsv", sep="\t")
    dev_data = pd.read_csv("./data/dev.tsv", sep="\t")

    train_data["sentence_length"] = train_data["sentence"].apply(lambda x: len(x))
    print(train_data.iloc[:, 2:])


if __name__ == "__main__":
    # dm01_label_sns_counterpoint()
    dm02_len_sns_counterpoint_disappoint()
