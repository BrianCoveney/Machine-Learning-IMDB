#
# Author: Brian Coveney
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
import itertools
from collections import Counter
from math import log, exp

# training datasets
path_pos = '/home/brian/Desktop/4thYr_MachineLearning/Assignment/IMDB/pos'
listing_pos = os.listdir(path_pos)

path_neg = '/home/brian/Desktop/4thYr_MachineLearning/Assignment/IMDB/neg'
listing_neg = os.listdir(path_neg)

# test datasets
path_pos_test = '/home/brian/Desktop/4thYr_MachineLearning/Assignment/smallTest/pos'
listing_pos_test = os.listdir(path_pos_test)


def inputNewDocument(path):
    dict = {}
    filename = os.path.join(path, '0_10.txt')
    with open(filename, 'r') as f:
        for line in f:
            for word in re.findall(r'[\w]+', line.lower()):
                dict[word] = dict.get(word, 0) + 1
    f.close()
    return dict


def readPosAndNegDocuments():
    vocab_dict = {}
    path = [path_pos, path_neg]
    for i in path:
        for filename in os.listdir(i):
            with open(os.path.join(i, filename), 'r', encoding='utf8') as f:
                for line in f:
                    for word in re.findall(r'[\w]+', line.lower()):
                        vocab_dict[word] = vocab_dict.get(word, 0) + 1

    return vocab_dict


def preProcessWords(vocab):
    vocab = {re.sub(r'[\W_]+', '', i) for i in vocab}
    return vocab


def createDictionary(listing, path):
    word_dict = {}
    for eachFile in listing:
        f = open(os.path.join(path, eachFile), "r", encoding='utf8')
        for line in f:
            for word in re.findall(r'[\w]+', line.lower()):
                word_dict[word] = word_dict.get(word, 0) + 1
    f.close()
    return word_dict


def calcProbability(wd_dict):
    words_type_dict = {}

    vocab = readPosAndNegDocuments()
    vocab_size = len(vocab)

    for k, v in wd_dict.items():

        # Calculate the conditional probabilities in the multinominal model:
        #
        # count(w,c) + 1 / count(c)+|V|
        #   count(w,c)  The num of occurrences of the word w in all documents of class c.
        #   count(c)    The total num of words in all documents of class c (incl duplicates).
        #   |V|         The number of words in the vocabulary
        #
        # Calc Prior Probabilities
        #   P(c) = num of documents of class c / total num of documents
        #
        num_word_occur_in_class_x = wd_dict[k]
        num_words_in_class_x = sum(wd_dict.values())
        conditional_prob = (num_word_occur_in_class_x + 1) / (num_words_in_class_x + vocab_size)

        words_type_dict[k] = conditional_prob


    return words_type_dict


def getNumDocsOfClassPos():
    #   Calc Prior Probabilities
    #   P(c) = num of documents of class c / total num of documents
    test_pos_doc_dict = inputNewDocument(path_pos_test)
    num_docs_of_c_pos = sum(test_pos_doc_dict.values())
    return num_docs_of_c_pos


def createPositiveWordDict():
    pos_wd_dict = createDictionary(listing_pos, path_pos)
    prob_pos_word = calcProbability(pos_wd_dict)
    print(prob_pos_word)
    return prob_pos_word


def createNegativeWordDict():
    neg_wd_dict = createDictionary(listing_neg, path_neg)
    prob_neg_word = calcProbability(neg_wd_dict)
    print(prob_neg_word)
    return prob_neg_word


def main():
    createPositiveWordDict()
    createNegativeWordDict()



if __name__ == "__main__":
    main()


# test datasets
path_pos_test = '/home/brian/Desktop/4thYr_MachineLearning/Assignment/smallTest/pos'
listing_pos_test = os.listdir(path_pos_test)