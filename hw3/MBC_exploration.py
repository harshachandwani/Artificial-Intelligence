import sys
import math
import copy
import random
import os

import sklearn
from sklearn import svm
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectPercentile
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_selection import chi2


def classify(train, test, train_data, test_data, ngram):

    #Naive Bayes Classifier
    classifier = MultinomialNB()
    classifier.fit(train_data, train.target)
    predict = classifier.predict(test_data)
    row = []
    row.append('Naive Bayes ' + ngram)
    calculate_metric(row, test, predict)
    res.append(row)

    classifier = RandomForestClassifier()
    classifier.fit(train_data, train.target)
    predict = classifier.predict(test_data)
    row = []
    row.append('Random Forest ' + ngram)
    calculate_metric(row, test, predict)
    res.append(row)

    classifier = LinearSVC()
    classifier.fit(train_data, train.target)
    predict = classifier.predict(test_data)
    row = []
    row.append('SVM ' + ngram)
    calculate_metric(row, test, predict)
    res.append(row)

    classifier = LogisticRegression()
    classifier.fit(train_data, train.target)
    predict = classifier.predict(test_data)
    row = []
    row.append('Logistic Regression ' + ngram)
    calculate_metric(row, test, predict)
    res.append(row)
    print(res)


def calculate_metric(row, test, pred):
    precision = metrics.precision_score(test.target, pred, average='macro')
    recall = metrics.recall_score(test.target, pred, average='macro')
    f1 = metrics.f1_score(test.target, pred, average='macro')
    row.append(precision)
    row.append(recall)
    row.append(f1)


def removeStopWords(data):
    stop_words = set(stopwords.words('english'))
    clean_data = copy.deepcopy(data)
    limit = len(clean_data.data)
    for i in range(limit):
        words = clean_data.data[i].split()
        removed_stop_words = []
        for w in words:
            if not w.lower() in stop_words:
                removed_stop_words.append(w)
        clean_data.data[i] = ' '.join(removed_stop_words)
    return clean_data


def stemmer(data):
    ps = PorterStemmer()
    stem_data = copy.deepcopy(data)
    limit = len(stem_data.data)
    for i in range(limit):
        stemmedWords = []
        data_words = stem_data.data[i].split()
        for word in data_words:
            stemmedWords.append(ps.stem(word))
        stem_data.data[i] = ' '.join(stemmedWords)
    return stem_data


def getBestConfig(train, test, output):
    #stem and stop
    train_stem = stemmer(train)
    test_stem = stemmer(test)
    train_stop = removeStopWords(train)
    test_stop = removeStopWords(test)
    train_stem_stop = stemmer(removeStopWords(train))
    test_stem_stop = stemmer(removeStopWords(test))
    train_nostem_nostop = copy.deepcopy(train)
    test_nostem_nostop = copy.deepcopy(test)

    #count_vect = CountVectorizer(ngram_range=(1,1), decode_error='ignore')
    #tfidf_vect = TfidfVectorizer(ngram_range=(1,1), decode_error='ignore')
    result = []

    classify('count_vect', train_stem, test_stem, result,
             "NB : Unigram: CountVectorizer: Stemmer: No Stopper")
    #classify(count_vect, train_stop, test_stop, result,"NB: Unigram: CountVectorizer: Stopper: No Stemmer")
    classify('count_vect', train_stem_stop, test_stem_stop, result,
             "NB: Unigram: CountVectorizer: Stemmer: Stopper")
    #classify(count_vect, train_nostem_nostop, test_nostem_nostop, result,"NB: Unigram: CountVectorizer: No Stemmer: No Stopper")
    #classify(count_vect, train_nostem_nostop, test_nostem_nostop, result,
    #         "NB : Unigram: CountVectorizer: No Stemmer: No Stopper: Feature:SelectPercentile = 80", True)
    #classify(count_vect, train_stem_stop, test_stem_stop, result,
    #             "NB : Unigram: CountVectorizer: Stemmer: Stopper: Feature:SelectPercentile = 80", True)
    classify(
        'count_vect', train_stem, test_stem, result,
        "NB : Unigram: CountVectorizer: Stemmer: No Stopper: Feature:SelectPercentile = 80",
        True)
    #classify(count_vect, train_stop, test_stop, result,
    #         "NB : Unigram: CountVectorizer: No Stemmer: Stopper: Feature:SelectPercentile = 80", True)

    classify('tfidf_vect', train_stem, test_stem, result,
             "NB: Unigram: TFIDFVectorizer: Stemmer: No Stopper")
    classify('tfidf_vect', train_stop, test_stop, result,
             "NB: Unigram: TFIDFVectorizer: Stopper: No Stemmer")
    #classify(tfidf_vect, train_stem_stop, test_stem_stop, result,"NB: Unigram: TFIDFVectorizer: Stemmer: Stopper")
    classify('tfidf_vect', train_nostem_nostop, test_nostem_nostop, result,
             "NB: Unigram: TFIDFVectorizer: No Stemmer: No Stopper")
    classify(
        'tfidf_vect', train_nostem_nostop, test_nostem_nostop, result,
        "NB : Unigram: TFIDFVectorizer: No Stemmer: No Stopper: Feature:SelectPercentile = 80",
        True)

    #classify(tfidf_vect, train_stem_stop, test_stem_stop, result,
    #        "NB : Unigram: TFIDFVectorizer: Stemmer: Stopper: Feature:SelectPercentile = 80", True)

    #classify(tfidf_vect, train_stem, test_stem, result,
    #         "NB : Unigram: TFIDFVectorizer: Stemmer: No Stopper: Feature:SelectPercentile = 80", True)

    classify(
        'tfidf_vect', train_stop, test_stop, result,
        "NB : Unigram: TFIDFVectorizer: No Stemmer: Stopper: Feature:SelectPercentile = 80",
        True)

    columns = ['Algorithm', 'Precision', 'Recall', 'F1 Value']

    from tabulate import tabulate
    print(tabulate(result, columns))
    print('\n\n')
    print(
        "Best Model is- NB : Unigram: TFIDFVectorizer: No Stemmer: No Stopper")

    fout = open(output, 'w')
    for row in result:
        fout.write(
            str(row[0]) + ',' + str(row[1]) + ',' + str(row[2]) + ',' +
            str(row[3]))
        fout.write('\n')
    fout.close()


def classify(vector, train, test, res, name, feature=False):
    if vector == 'count_vect':
        vect = CountVectorizer(ngram_range=(1, 1), decode_error='ignore')
    else:
        vect = TfidfVectorizer(ngram_range=(1, 1), decode_error='ignore')

    train_data = vect.fit_transform(train.data)
    test_data = vect.transform(test.data)
    if feature:
        chi_2 = SelectPercentile(chi2, percentile=85)
        feature_train = chi_2.fit_transform(train_data, train.target)
        feature_test = chi_2.transform(test_data)
        classifier = MultinomialNB(alpha=0.005)
        classifier.fit(feature_train, train.target)
        predict = classifier.predict(feature_test)
    else:
        classifier = MultinomialNB(alpha=0.005)
        classifier.fit(train_data, train.target)
        predict = classifier.predict(test_data)

    algo = name
    row = []
    row.append(algo)
    calculate_metric(row, test, predict)
    res.append(row)
    return classifier


def calculate_metric(row, test, pred):
    precision = metrics.precision_score(test.target, pred, average='macro')
    recall = metrics.recall_score(test.target, pred, average='macro')
    f1 = metrics.f1_score(test.target, pred, average='macro')
    row.append(precision)
    row.append(recall)
    row.append(f1)


if __name__ == '__main__':

    if len(sys.argv) == 4:
        trainset = sys.argv[1]
        testset = sys.argv[2]
        output = sys.argv[3]

    else:
        print("Please enter all valid arguments")
        print("python UB_BB.py trainset testset output")
        sys.exit(1)

    training_data = load_files(trainset)
    testing_data = load_files(testset)

    #Remove headers
    for i in range(len(training_data.data)):
        lines = training_data.data[i].splitlines()
        for j, line in enumerate(lines):
            if ':' not in str(line):
                training_data.data[i] = str(b'\n'.join(lines[j + 1:]))
                break

    for i in range(len(testing_data.data)):
        lines = testing_data.data[i].splitlines()
        for j, line in enumerate(lines):
            if ':' not in str(line):
                testing_data.data[i] = str(b'\n'.join(lines[j + 1:]))
                break

    getBestConfig(training_data, testing_data, output)
