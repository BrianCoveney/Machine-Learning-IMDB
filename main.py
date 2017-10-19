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
from math import log

path_pos = '/home/brian/Desktop/4thYr_MachineLearning/Assignment/smallTest/pos'
listing_pos = os.listdir(path_pos)

path_neg = '/home/brian/Desktop/4thYr_MachineLearning/Assignment/smallTest/neg'
listing_neg = os.listdir(path_neg)


def readPosAndNegDocuments():
    vocab = set([])
    path = [path_pos, path_neg]
    for i in path:
        for filename in os.listdir(i):
            with open(os.path.join(i, filename), 'r') as f:
                words = f.read().lower().split()
                vocab.update(words)

    vocab_dict = dict.fromkeys(vocab, 0)

    return vocab_dict


# Currently un-used
def preProcessWords(vocab):
    vocab = {re.sub(r'[\W_]+', '', i) for i in vocab}
    return vocab


def createDictionary(listing, path):
    word_dict = {}
    for eachFile in listing:
        f = open(path + "/" + eachFile, "r", encoding='utf8')
        for line in f:
            for word in re.findall(r'[\w]+', line.lower()):
                word_dict[word] = word_dict.get(word, 0) + 1
    f.close()
    return word_dict


def calcProbability(wd_dict):
    words_type_dict = {}
    vocab = readPosAndNegDocuments()
    for k, v in wd_dict.items():

        # Variables
        vocab_size = sum(vocab.values())
        num_word_occur_in_class_x = wd_dict[k]
        num_words_in_class_x = sum(wd_dict.values())

        # Calculate the conditional probabilities in the multinominal model
        words_type_dict[k] = num_word_occur_in_class_x + 1 / num_words_in_class_x + vocab_size

    return words_type_dict


def createPositiveWordDict():
    pos_wd_dict = createDictionary(listing_pos, path_pos)
    print('pos dict len', len(pos_wd_dict))

    prob_pos_word = calcProbability(pos_wd_dict)
    print("Positive", prob_pos_word, "\n")


def createNegativeWordDict():
    neg_wd_dict = createDictionary(listing_neg, path_neg)
    print('neg dict len', len(neg_wd_dict))

    prob_neg_word = calcProbability(neg_wd_dict)
    print("Negative", prob_neg_word)


def main():
    createPositiveWordDict()
    createNegativeWordDict()


if __name__ == "__main__":
    main()
