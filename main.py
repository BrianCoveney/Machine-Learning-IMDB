import os
from collections import Counter

path_neg = '/home/brian/Desktop/4thYr_MachineLearning/Assignment/LargeIMDB/neg'

listing = os.listdir(path_neg)


def read_words(list_files):

    vocab_set = set()

    for eachFile in list_files:
        f = open(path_neg + "/" + eachFile, "r")
        words = f.read()
        f.close()

        vocab_set.add(words)

    return vocab_set


def print_words():
    words = read_words(listing)

    for w in words:
        print(w)


def main():

    read_words(listing)

    print_words()


if __name__ == "__main__":
    main()
