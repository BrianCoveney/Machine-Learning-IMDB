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
    vocab = set([])
    path = [path_pos, path_neg]
    for i in path:
        for filename in os.listdir(i):
            with open(os.path.join(i, filename), 'r') as f:
                words = f.read().lower().split()
                vocab.update(words)

    cleaned_vocab = preProcessWords(vocab)

    vocab_dict = dict.fromkeys(cleaned_vocab, 0)

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
    prob_of_class_x = 0

    for k, v in wd_dict.items():
        vocab_size = sum(vocab.values())
        num_word_occur_in_class_x = wd_dict[k]
        num_words_in_class_x = sum(wd_dict.values())

        # Calculate the conditional probabilities in the multinominal model:
        # count(w,c) + 1 / count(c)+|V|
        words_type_dict[k] = num_word_occur_in_class_x + 1 / num_words_in_class_x + vocab_size

        # Sum of  log𝑃(𝑤|𝑐)
        prob_of_class_x += log(words_type_dict[k])

    return prob_of_class_x


def testDocumentWithPositive():
    log_p_of_pos = createPositiveWordDict()
    t_dict = inputNewDocument(path_pos_test)
    for k, v in t_dict.items():
        t_dict[k] = log_p_of_pos
    print('Pos', list(t_dict.values())[0])


def testDocumentWithNegative():
    log_p_of_neg = createNegativeWordDict()
    t_dict = inputNewDocument(path_pos_test)
    for k, v in t_dict.items():
        t_dict[k] = log_p_of_neg
    print('Neg', list(t_dict.values())[0])


def createPositiveWordDict():
    pos_wd_dict = createDictionary(listing_pos, path_pos)
    prob_pos_word = calcProbability(pos_wd_dict)
    return prob_pos_word


def createNegativeWordDict():
    neg_wd_dict = createDictionary(listing_neg, path_neg)
    prob_neg_word = calcProbability(neg_wd_dict)
    return prob_neg_word


def main():
    createPositiveWordDict()
    createNegativeWordDict()

    testDocumentWithPositive()
    testDocumentWithNegative()


if __name__ == "__main__":
    main()


# test datasets
path_pos_test = '/home/brian/Desktop/4thYr_MachineLearning/Assignment/smallTest/pos'
listing_pos_test = os.listdir(path_pos_test)