#
# Author:   Brian Coveney
# Date:     October 2017
#
# Naive Bayes's Theorem
#
#   Calc Prior Probabilities
#   P(c) = num of documents of class c / total num of documents
#
#   Naïve Bayes - Multinomial Model
#   P(w|c) = count(w,c) + 1 / count(c)+|V|
#
#   count(w,c)  The num of occurrences of the word w in all documents of class c.
#   count(c)    The total num of words in all documents of class c (incl duplicates).
#   |V|         The number of words in the vocabulary
#

import os
import re
from math import log
import numpy as np

# training datasets
path_pos = '/home/brian/Desktop/4thYr_MachineLearning/Assignment/LargeIMDB/pos'
listing_pos = os.listdir(path_pos)

path_neg = '/home/brian/Desktop/4thYr_MachineLearning/Assignment/LargeIMDB/neg'
listing_neg = os.listdir(path_neg)

# test dataset
path_pos_test = '/home/brian/Desktop/4thYr_MachineLearning/Assignment/smallTest/pos'
listing_pos_test = os.listdir(path_pos_test)

path_neg_test = '/home/brian/Desktop/4thYr_MachineLearning/Assignment/smallTest/neg'
listing_neg_test = os.listdir(path_neg_test)


path_test_parent = '/home/brian/Desktop/4thYr_MachineLearning/Assignment/smallTest/'

path_type_pos = 'pos'
doc_to_test_pos = '14_8.txt'

path_type_neg = 'neg'
doc_to_test_neg = '14_1.txt'


# Iterate over all documents and words in both directories 'pos' and 'neg'.
# Then returns the 'vocab' as a dictionary containing words(key) and word-frequency(value)
def readPosAndNegDocuments(path_p, path_n):
    vocab_pos_and_neg_dict = {}
    path = [path_p, path_n]
    for i in path:
        for filename in os.listdir(i):
            with open(os.path.join(i, filename), 'r', encoding='utf8') as f:
                for line in f:
                    for word in re.findall(r'[\w]+', line.lower()):
                        vocab_pos_and_neg_dict[word] = vocab_pos_and_neg_dict.get(word, 0) + 1
            f.close()
    return vocab_pos_and_neg_dict


# Iterate over the 'pos' test document.
# Then return a dictionary with words(key) and word-frequency(value)
def inputNewDocument(path_parent, path_type, doc):
    test_new_doc_dict = {}
    filename = os.path.join(path_parent, path_type, doc)
    with open(filename, 'r') as f:
        for line in f:
            for word in re.findall(r'[\w]+', line.lower()):
                test_new_doc_dict[word] = test_new_doc_dict.get(word, 0) + 1
    f.close()
    return test_new_doc_dict


# Pass in either a 'pos' or 'neg' directory.
# Iterate over all documents and words in that directory.
# Then return a dictionary with words(key) and word-frequency(value)
def createDictionaryForClass(listing, path):
    word_dict = {}
    for eachFile in listing:
        f = open(os.path.join(path, eachFile), "r", encoding='utf8')
        for line in f:
            for word in re.findall(r'[\w]+', line.lower()):
                word_dict[word] = word_dict.get(word, 0) + 1

    f.close()
    return word_dict


# Create a 'pos' dictionary
# Then create a dictionary for the positive conditional probabilities
def getPosConditionalProbabilitiesDict():
    pos_wd_dict = createDictionaryForClass(listing_pos, path_pos)
    pos_words_dict = calcConditionalProbabilities(pos_wd_dict)

    ###--------------------DEBUG STATEMENTS----------------------
    #
    # print(pos_words_dict)
    #
    # e.g.  'bad': 0.0004860267314702309,
    #
    ###--------------------DEBUG STATEMENTS----------------------

    return pos_words_dict


def getPosTestDocConditionalProbabilities():
    pos_test_dict = createDictionaryForClass(listing_pos_test, path_pos_test)
    pos_test_words_dict = calcConditionalProbabilities(pos_test_dict)
    return pos_test_words_dict


def getNegTestDocConditionalProbabilities():
    neg_test_dict = createDictionaryForClass(listing_neg_test, path_neg_test)
    neg_test_words_dict = calcConditionalProbabilities(neg_test_dict)
    return neg_test_words_dict


# Create a 'neg' dictionary
# Then create a dictionary for the negative conditional probabilities
def getNegConditionalProbabilitiesDict():
    neg_wd_dict = createDictionaryForClass(listing_neg, path_neg)
    neg_words_dict = calcConditionalProbabilities(neg_wd_dict)

    ###--------------------DEBUG STATEMENTS----------------------
    #
    # print(neg_words_dict)
    #
    # e.g.  'bad': 0.0015408802222440575,
    #
    ###--------------------DEBUG STATEMENTS----------------------

    # for k, v in neg_words_dict.items():

    return neg_words_dict


def preProcessWords(vocab):
    vocab = {re.sub(r'[\W_]+', '', i) for i in vocab}
    print(type(vocab))
    return vocab


# Pass in either a 'pos' or 'neg' directory.
def calcConditionalProbabilities(wd_dict):
    words_type_dict = {}
    vocab = readPosAndNegDocuments(path_pos, path_neg)
    vocab_size = len(vocab)  # |V|

    classification = 0
    for k, v in wd_dict.items():
        num_word_occur_in_class_x = wd_dict[k]        # count(w,c)
        num_words_in_class_x = sum(wd_dict.values())  # count(c)

        # Naïve Bayes - Multinomial Model:
        # P(w|c) = count(w,c) + 1 / count(c)+|V|
        conditional_prob = (num_word_occur_in_class_x + 1) / (num_words_in_class_x + vocab_size)

        words_type_dict[k] = conditional_prob

        classification += words_type_dict[k]

    return words_type_dict, classification


def getMeanProb(dict_to_test, dict_type, conditionalProbType):
    total_num_docs = 25000
    count = 0
    list_prob = []
    for key, val in dict_to_test.items():
        if key in dict_type.keys():
            count += 1
        prior_prob = log(count / total_num_docs)
        classify = prior_prob + log(conditionalProbType)

        list_prob.append(classify)

    numpy_array = np.array(list_prob)
    mean = np.mean(numpy_array)
    return mean


def testSinglePosDocument():
    dict_doc_to_test = inputNewDocument(path_test_parent, path_type_pos, doc_to_test_pos)

    print('File:', path_test_parent + path_type_pos + '/' + doc_to_test_pos)

    pos_dict, sum_of_pos_prob = getPosConditionalProbabilitiesDict()
    pos_mean = getMeanProb(dict_doc_to_test, pos_dict, sum_of_pos_prob)
    print("Score(positive)  :", pos_mean)

    neg_dict, sum_of_neg_prob = getNegConditionalProbabilitiesDict()
    neg_mean = getMeanProb(dict_doc_to_test, neg_dict, sum_of_neg_prob)
    print("Score(negative)  :", neg_mean)

    if pos_mean > neg_mean:
        print('Classify document as positive')
    else:
        print('Classify document as negative')

    ###--------------------DEBUG STATEMENTS----------------------
    #
    # File: pos / 1_10.txt
    # Score(positive) : -6.51499977775
    # Score(negative) : -6.51561472337
    # Classify document as positive
    #
    ###--------------------DEBUG STATEMENTS----------------------


def testSingleNegDocument():
    dict_doc_to_test = inputNewDocument(path_test_parent, path_type_neg, doc_to_test_neg)

    print('File:', path_test_parent + path_type_neg + '/' + doc_to_test_neg)

    pos_dict, sum_of_pos_prob = getPosConditionalProbabilitiesDict()
    pos_mean = getMeanProb(dict_doc_to_test, pos_dict, sum_of_pos_prob)
    print("Score(positive)  :", pos_mean)

    neg_dict, sum_of_neg_prob = getNegConditionalProbabilitiesDict()
    neg_mean = getMeanProb(dict_doc_to_test, neg_dict, sum_of_neg_prob)
    print("Score(negative)  :", neg_mean)

    if pos_mean > neg_mean:
        print('Classify document as positive')
    else:
        print('Classify document as negative')


    ###--------------------DEBUG STATEMENTS----------------------
    #
    # File: pos / 1_10.txt
    # Score(positive) : -6.50959402626
    # Score(negative) : -6.5039541065
    # Classify document as negative
    #
    ###--------------------DEBUG STATEMENTS----------------------


def testAllPosDocuments():
    print('Directory: pos')
    test_pos_dict = createDictionaryForClass(listing_pos_test, path_pos_test)

    pos_dict, sum_of_pos_prob = getPosConditionalProbabilitiesDict()
    pos_mean = getMeanProb(test_pos_dict, pos_dict, sum_of_pos_prob)
    print("Score(positive)  :", pos_mean)

    neg_dict, sum_of_neg_prob = getNegConditionalProbabilitiesDict()
    neg_mean = getMeanProb(test_pos_dict, neg_dict, sum_of_neg_prob)
    print("Score(negative)  :", neg_mean)

    if pos_mean > neg_mean:
        print('Classify document as positive')
    else:
        print('Classify document as negative')

    ###--------------------DEBUG STATEMENTS----------------------
    #
    # Directory: pos
    # Score(positive) : -1.51376662543
    # Score(negative) : -1.51658725535
    # Classify document as positive
    #
    ###--------------------DEBUG STATEMENTS----------------------


def testAllNegDocuments():
    print('Directory: neg')
    test_neg_dict = createDictionaryForClass(listing_neg_test, path_neg_test)

    pos_dict, sum_of_pos_prob = getPosConditionalProbabilitiesDict()
    pos_mean = getMeanProb(test_neg_dict, pos_dict, sum_of_pos_prob)
    print("Score(positive)  :", pos_mean)

    neg_dict, sum_of_neg_prob = getNegConditionalProbabilitiesDict()
    neg_mean = getMeanProb(test_neg_dict, neg_dict, sum_of_neg_prob)
    print("Score(negative)  :", neg_mean)

    if pos_mean > neg_mean:
        print('Classify document as positive')
    else:
        print('Classify document as negative')

    ###--------------------DEBUG STATEMENTS----------------------
    #
    # Directory: pos
    # Score(positive) : -1.50750295311
    # Score(negative) : -1.50064233133
    # Classify document as negative
    #
    ###--------------------DEBUG STATEMENTS----------------------


def main():
    # testSingleDocument()
    # testAllPosDocuments()
    # testAllNegDocuments()
    testSingleNegDocument()

if __name__ == "__main__":
    main()
