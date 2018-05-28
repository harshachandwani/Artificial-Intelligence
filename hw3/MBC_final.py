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
import pickle

if __name__ == '__main__':

    if len(sys.argv) == 4:
        trainset = sys.argv[1]
        testset = sys.argv[2]
        output = sys.argv[3]

    else:
        print("Please enter all valid arguments")
        print("python MBC_final.py trainset testset output")
        sys.exit(1)

    training_data = load_files(trainset)
    testing_data = load_files(testset)

    #Remove headers
    limit = len(training_data.data)
    for i in range(limit):
        lines = training_data.data[i].splitlines()
        for j, line in enumerate(lines):
            if ':' not in str(line):
                training_data.data[i] = str(b'\n'.join(lines[j + 1:]))
                break

    limit = len(testing_data.data)
    for i in range(limit):
        lines = testing_data.data[i].splitlines()
        for j, line in enumerate(lines):
            if ':' not in str(line):
                testing_data.data[i] = str(b'\n'.join(lines[j + 1:]))
                break

    vect = TfidfVectorizer(ngram_range=(1, 1), decode_error='ignore')
    train_data = vect.fit_transform(training_data.data)
    test_data = vect.transform(testing_data.data)
    '''
    chi_2 = SelectPercentile(chi2, percentile=80)
    feature_train = chi_2.fit_transform(train_data, training_data.target)
    feature_test = chi_2.transform(test_data)
    '''
    classifier = MultinomialNB(alpha=0.005)
    classifier.fit(train_data, training_data.target)
    predict = classifier.predict(test_data)

    f1 = metrics.f1_score(testing_data.target, predict, average='macro')
    precision = metrics.precision_score(
        testing_data.target, predict, average='macro')
    recall = metrics.recall_score(testing_data.target, predict, average='macro')
    print("F1 = " + str(f1))
    print("Precision = " + str(precision))
    print("Recall = " + str(recall))

    fout = open(output, 'w')
    fout.write(str(f1))
    fout.write('\n')
    fout.close()
