import pandas as pd
import numpy as np
import jieba
from gensim import corpora, models
import jieba.posseg as jp, jieba

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics


def cut(column):
    stopwords = [line.strip() for line in open('stop_words.txt', 'r', encoding='utf-8').readlines()]

    def is_Chinese(ch):
        if '\u4e00' <= ch <= '\u9fff':
            return True
        return False

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
        return False

    str1 = str(column)
    strlist = jieba.lcut(str1)
    result = ''
    for word in strlist:

        if word == '\ufeff':
            continue
        elif (is_Chinese(word) or is_number(word)) and word not in stopwords:
            result = result + word + ' '
        else:
            continue
    result = result.rstrip()
    return result


def get_data():
    df = pd.read_csv('online_shopping_10_cats.csv')
    df['words'] = df['review'].map(lambda i : cut(i))
    df.to_csv('review_splited.csv')


def get_TF_IDF(data):
    words = data['words'].apply(lambda x: np.str_(x))

    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(
        vectorizer.fit_transform(words))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    return weight, data['label']


def get_LDA(data):
    words_ls = []

    for words in data['words'].apply(lambda x: np.str_(x)):
        # t = [i for i in words.split(' ')]
        words_ls.append(words.split(' '))

    dictionary = corpora.Dictionary(words_ls)
    # 基于词典，使【词】→【稀疏向量】，并将向量放入列表，形成【稀疏向量集】
    corpus = [dictionary.doc2bow(words) for words in words_ls]
    # lda模型，num_topics设置主题的个数
    lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)

    return lda.inference(corpus)[0], data['label']



def train_and_evaluate(x, y, cat, model=linear_model.LogisticRegression):
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.4, random_state=True)

    # reg = linear_model.LinearRegression()
    reg = linear_model.LogisticRegression()
    reg.fit(train_X, train_y)

    y = reg.predict(test_X)
    y = [1 if i > 0.5 else 0 for i in y]
    yy = test_y.to_list()

    print("%s\t\taccuracy: %f\t\trecall: %f\t\tf1 score: %f\n" % (cat, metrics.accuracy_score(yy, y),
                                  metrics.recall_score(yy, y),
                                  metrics.f1_score(yy, y)))


def TF_IDF():
    df = pd.read_csv('data.csv')

    categories = df['cat'].unique()

    for cat in categories:
        data = df[df['cat'] == cat]

        x, y = get_TF_IDF(data)
        # y = data['label']
        train_and_evaluate(x, y, cat)


def LDA():
    df = pd.read_csv('review_splited.csv')

    categories = df['cat'].unique()

    for cat in categories:
        data = df[df['cat'] == cat]

        x, y = get_LDA(data)
        train_and_evaluate(x, y, cat)


if __name__ == '__main__':
    # get_data()
    # LDA()
    TF_IDF()


