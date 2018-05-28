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
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

res = []


def classify(train, test, train_data, test_data, ngram):

    #Naive Bayes Classifier
    classifier = MultinomialNB()
    classifier.fit(train_data, train.target)
    predict = classifier.predict(test_data)
    row = []
    row.append('NB,' + ngram)
    calculate_metric(row, test, predict)
    res.append(row)

    classifier = RandomForestClassifier()
    classifier.fit(train_data, train.target)
    predict = classifier.predict(test_data)
    row = []
    row.append('RF,' + ngram)
    calculate_metric(row, test, predict)
    res.append(row)

    classifier = LinearSVC()
    classifier.fit(train_data, train.target)
    predict = classifier.predict(test_data)
    row = []
    row.append('SVM,' + ngram)
    calculate_metric(row, test, predict)
    res.append(row)

    classifier = LogisticRegression()
    classifier.fit(train_data, train.target)
    predict = classifier.predict(test_data)
    row = []
    row.append('LR,' + ngram)
    calculate_metric(row, test, predict)
    res.append(row)


def calculate_metric(row, test, predict):
    precision = metrics.precision_score(test.target, predict, average='macro')
    recall = metrics.recall_score(test.target, predict, average='macro')
    f1 = metrics.f1_score(test.target, predict, average='macro')
    row.append(precision)
    row.append(recall)
    row.append(f1)


def plot_graph(train, test, uni_train_data, uni_test_data):
    data_siz = uni_train_data.shape[0]
    step = 200
    print('Plotting the learning curve')
    graph_data = {}

    if (data_siz % step != 0):
        limit = int(data_siz / step + 1)
    else:
        limit = int(data_siz / step)

    for i in range(limit):
        if step > data_siz:
            curr_siz = data_siz
        else:
            step += 200
            curr_siz = step
        j = 1

        while (j <= 4):
            if (j == 1):
                classifier = MultinomialNB()
                algo = 'Naive Bayes unigram'
            if (j == 2):
                classifier = LinearSVC()
                algo = 'SVM unigram'
            if (j == 3):
                classifier = RandomForestClassifier()
                algo = 'Random Forest unigram'
            if (j == 4):
                classifier = LogisticRegression()
                algo = 'Logistic Regression unigram'

            curr_siz = int(curr_siz)
            classifier.fit(uni_train_data[:curr_siz], train.target[:curr_siz])
            predict = classifier.predict(uni_test_data)
            f1 = metrics.f1_score(test.target, predict, average='macro')
            graph_data.setdefault(algo, {'data_sizes': [], 'f1_values': []})
            graph_data[algo]['data_sizes'].append(curr_siz)
            graph_data[algo]['f1_values'].append(f1)
            j += 1

    step = 200
    fig = plt.figure()
    clrs = ['red', 'blue', 'green', 'violet']
    plt.title('Learning curves Comparison')
    plt.ylim(0.3, 1)
    plt.xlabel("train set size Step size = 200")
    plt.ylabel("F-1 Value")
    plt.grid()

    for algo, clr in zip(sorted(graph_data), clrs):
        sizes = graph_data[algo]['data_sizes']
        f1_scores = graph_data[algo]['f1_values']
        col = colors.cnames[clr]
        plt.plot(
            sizes,
            f1_scores,
            linestyle='dashed',
            linewidth=3,
            marker='o',
            color=col,
            label=algo)

    lgd = plt.legend(bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout()
    plt.show(lgd)
    fig.savefig('LC_plot', bbox_inches='tight')


if __name__ == '__main__':

    if len(sys.argv) == 5:
        trainset = sys.argv[1]
        testset = sys.argv[2]
        output = sys.argv[3]
        display_LC = sys.argv[4]

    else:
        print("Please enter all valid arguments")
        print("python UB_BB.py trainset testset output display_LC")
        sys.exit(1)

    training_data = load_files(trainset)
    testing_data = load_files(testset)

    #Remove headers
    limit = len(training_data.data)
    for i in range(limit):
        lines = training_data.data[i].splitlines()
        for j, line in enumerate(lines):
            if ':' not in str(line):
                nonHeaders = str(b'\n'.join(lines[j + 1:]))
                training_data.data[i] = nonHeaders
                break

    limit = len(testing_data.data)
    for i in range(limit):
        lines = testing_data.data[i].splitlines()
        for j, line in enumerate(lines):
            if ':' not in str(line):
                nonHeaders = str(b'\n'.join(lines[j + 1:]))
                testing_data.data[i] = nonHeaders
                break

    vect = CountVectorizer(ngram_range=(1, 1), decode_error='ignore')
    train_data = vect.fit_transform(training_data.data)
    test_data = vect.transform(testing_data.data)
    classify(training_data, testing_data, train_data, test_data, 'UB')

    #to be used later while graph plot
    uni_train_data = train_data
    uni_test_data = test_data

    vect = CountVectorizer(ngram_range=(1, 2), decode_error='ignore')
    train_data = vect.fit_transform(training_data.data)
    test_data = vect.transform(testing_data.data)
    classify(training_data, testing_data, train_data, test_data, 'BB')

    result = []
    for i in range(int(len(res) / 2)):
        result.append(res[i])
        result.append(res[i + 4])

    fout = open(output, 'w')
    for row in result:
        fout.write(
            str(row[0]) + ',' + str(row[1]) + ',' + str(row[2]) + ',' +
            str(row[3]))
        fout.write('\n')
    fout.close()
    if display_LC == 1:
        print("option check")
        plot_graph(training_data, testing_data, uni_train_data, uni_test_data)

    columns = ['Algorithm', 'Precision', 'Recall', 'F1 Value']
    from tabulate import tabulate
    print(tabulate(result, columns))
    print('\n\n')

    if display_LC == '1':
        plot_graph(training_data, testing_data, uni_train_data, uni_test_data)
