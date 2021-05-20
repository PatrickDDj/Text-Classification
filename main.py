import pandas as pd
import numpy as np
import jieba

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def cut(column):
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
        elif is_Chinese(word) or is_number(word):
            result = result + word + ' '
        else:
            continue
    result = result.rstrip()
    return result


if __name__ == '__main__':

    df = pd.read_csv('data.csv')

    categories = df['cat'].unique()

    for cat in categories:

        data = df[df['cat'] == cat]

        words = data['words'].apply(lambda x: np.str_(x))

        vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
        transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
        tfidf = transformer.fit_transform(
            vectorizer.fit_transform(words))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
        word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
        weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重

        train_X, test_X, train_y, test_y = train_test_split(weight, data['label'] , test_size=0.4, random_state=True)

        from sklearn import linear_model
        reg = linear_model.LinearRegression()
        # reg = linear_model.LogisticRegression()
        reg.fit(train_X, train_y)

        y = reg.predict(test_X)
        y = [1 if i>0.5 else 0 for i in y]
        yy = test_y.to_list()

        from sklearn import metrics
        # print("'%s'  Accuracy: %f,  Recall: %f,  F1: %f\n" % (cat, metrics.accuracy_score(y, yy),
        #                                                   metrics.recall_score(y,yy),
        #                                                   metrics.f1_score(y, yy)))
        print("%s\t\t%f\t%f\t%f\n" % (cat, metrics.accuracy_score(y, yy),
                                                              metrics.recall_score(y, yy),
                                                              metrics.f1_score(y, yy)))











