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
from math import log, exp

# training datasets
path_pos = '/home/brian/Desktop/4thYr_MachineLearning/Assignment/LargeIMDB/pos'
listing_pos = os.listdir(path_pos)

path_neg = '/home/brian/Desktop/4thYr_MachineLearning/Assignment/LargeIMDB/neg'
listing_neg = os.listdir(path_neg)

# test dataset
path_type = 'pos'
doc_to_test = '0_10.txt'
path_pos_test = '/home/brian/Desktop/4thYr_MachineLearning/Assignment/smallTest'
listing_pos_test = os.listdir(path_pos_test)


# Iterate over the 'pos' test document.
# Then return a dictionary with words(key) and word-frequency(value)
def inputNewDocument(path):
    test_new_doc_dict = {}
    filename = os.path.join(path, path_type, doc_to_test)
    with open(filename, 'r') as f:
        for line in f:
            for word in re.findall(r'[\w]+', line.lower()):
                test_new_doc_dict[word] = test_new_doc_dict.get(word, 0) + 1
    f.close()
    return test_new_doc_dict


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
    pos_words_dict = createDictionaryForClass(listing_pos, path_pos)
    pos_words_dict = calcConditionalProbabilities(pos_words_dict)

    ###--------------------DEBUG STATEMENTS----------------------
    #
    # print(pos_words_dict)
    #
    # e.g.  'bad': 0.0004860267314702309,
    #
    ###--------------------DEBUG STATEMENTS----------------------

    return pos_words_dict


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

        ###--------------------DEBUG STATEMENTS----------------------
        #
        # print(log(words_type_dict[k]))
        #
        # e.g. -12.422278855708885
        #
        ###--------------------DEBUG STATEMENTS----------------------

        # (Class | word) = logP(c) + logP(w|c) + logP(w|c) ... = x
        #
        # logP(w|c) + logP(w|c) + logP(w|c) + ...
        classification += words_type_dict[k]

    return words_type_dict, classification


def testDocument():
    pos_dict, sum_of_pos_prob = getPosConditionalProbabilitiesDict()
    neg_dict, sum_of_neg_prob = getNegConditionalProbabilitiesDict()
    test_pos_doc_dict = inputNewDocument(path_pos_test)

    ###--------------------DEBUG STATEMENTS----------------------
    #
    # print("class pos", sum_of_pos_prob)
    # print("class neg", sum_of_neg_prob)
    #
    # class pos -361968.3318638592
    # class neg -336494.0818642115
    #
    ###--------------------DEBUG STATEMENTS----------------------

    total_num_docs = 2000
    count_pos = 0
    list_pos = []
    for key, val in test_pos_doc_dict.items():
        if key in pos_dict.keys():
            count_pos += 1

        # Here we find the total amount of words from our test document
        # that are in the positive dictionary, and increment a counter.
        # Then we divide this by the total number of documents.
        #
        # 𝑃(𝑐) = Number of documents of class c / Total number of documents
        #
        # (Class | word) = logP(c) + logP(w|c) + logP(w|c) ... = x
        # logP(c)
        prior_prob_pos = log(count_pos / total_num_docs)
        classify_pos = prior_prob_pos + log(sum_of_pos_prob)

        ###--------------------DEBUG STATEMENTS----------------------
        #
        # print(Pos =>, key, " : ", exp(classify_pos))
        #
        # e.g.  Pos => good  :  0.0222
        #
        ###--------------------DEBUG STATEMENTS----------------------

        list_pos.append(classify_pos)

    sum_pos = sum(list_pos)
    print("Score(positive)  :", sum_pos)

    count_neg = 0
    list_neg = []
    for k, v in test_pos_doc_dict.items():
        if k in neg_dict.keys():
            count_neg += 1

        prior_prob_neg = log(count_neg / total_num_docs)
        classify_neg = prior_prob_neg + log(sum_of_neg_prob)

        ###--------------------DEBUG STATEMENTS--------------------
        #
        # print('Neg =>', k, " : ", exp(classify_neg))
        #
        # e.g.  Neg = > good: 0.0221
        #
        ###--------------------DEBUG STATEMENTS--------------------

        list_neg.append(classify_neg)

    sum_neg = sum(list_neg)
    print("Score(negative)  :", sum_neg)

    if sum_pos > sum_neg:
        print('Classify document as positive')
    else:
        print('Classify document as negative')

    ###--------------------DEBUG STATEMENTS--------------------
    #
    # Score(positive)  : -390.94857107767393
    # Score(negative)  : -391.0088357483707
    # Classify document as positive
    #
    ###--------------------DEBUG STATEMENTS--------------------


def main():
    # getPosConditionalProbabilitiesDict()
    # getNegConditionalProbabilitiesDict()

    testDocument()


if __name__ == "__main__":
    main()
