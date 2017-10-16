import os
import re

path_pos = '/home/brian/Desktop/4thYr_MachineLearning/Assignment/smallTest/pos'
listing_pos = os.listdir(path_pos)
path_neg = '/home/brian/Desktop/4thYr_MachineLearning/Assignment/smallTest/neg'
listing_neg = os.listdir(path_neg)

word_dict = {}
word_pos_dict = {}
word_neg_dict = {}


def readWords(listing, path):
    for eachFile in listing:
        f = open(path + "/" + eachFile, "r", encoding='utf8')
        for line in f:
            for word in re.findall(r'[\w]+', line.lower()):
                word_dict[word] = word_dict.get(word, 0.0) + 1.0

    return word_dict


def calcProbability(wd_dict, words_type_dict):
    for k, v in wd_dict.items():
        # Calculate the prior probabilities
        words_type_dict[k] = wd_dict[k] / sum(wd_dict.values())

        # Todo: Calculations of the probabilities in the multinominal model using laplace smoothing

    return word_pos_dict


def createPositiveWordDict(listings, path):
    pos_wd_dict = readWords(listings, path)
    prob_pos_word = calcProbability(pos_wd_dict, word_pos_dict)
    print("Positive", prob_pos_word, "\n")


def createNegativeWordDict(listings, path):
    neg_wd_dict = readWords(listings, path)
    prob_neg_word = calcProbability(neg_wd_dict, word_neg_dict)
    print("Negative", prob_neg_word)


def main():
    createPositiveWordDict(listing_pos, path_pos)
    createNegativeWordDict(listing_neg, path_neg)


if __name__ == "__main__":
    main()
