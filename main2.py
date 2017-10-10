import os
from collections import Counter

path_neg = '/home/brian/Desktop/4thYr_MachineLearning/Assignment/LargeIMDB/neg'

listing = os.listdir(path_neg)


def read_words(list_files):

    vocab_list = []

    for eachFile in list_files:
        f = open(path_neg + "/" + eachFile, "r")
        words = f.read().split()
        f.close()

        vocab_list.append(words)

    return vocab_list


def print_words():
    words = read_words(listing)
    print(words)
    # 12500
    print(len(words))

    # for w in words:
    #     freq = Counter(w)
    #     print(freq)
    #     print(w)


def main():

    read_words(listing)

    print_words()


if __name__ == "__main__":
    main()