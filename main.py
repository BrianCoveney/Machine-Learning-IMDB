import os
import string
import re
from collections import Counter
from string import digits

path_neg = '/home/brian/Desktop/4thYr_MachineLearning/Assignment/LargeIMDB/neg'
listing_neg = os.listdir(path_neg)


def read_words(list_files):
    vocab = set([])

    for eachFile in list_files:
        f = open(path_neg + "/" + eachFile, "r")
        words = remove_digits(f)
        words_copy = [x.lower() for x in words]
        f.close()
        vocab.update(words_copy)

    vocab_copy = remove_punt(vocab)

    print(vocab_copy)
    return vocab_copy


def remove_digits(file):
     return file.read().replace('0', '').replace('1', '').replace('2', '').replace('3', '').replace('4', '')\
         .replace('5', '').replace('6', '').replace('7', '').replace('8', '').replace('9', '').split()


def word_freq(vocab):

    count = {}

    for word in vocab:
        if word in count:
            count[word] += 1
        else:
            count[word] = 1

    neg_dict = dict.fromkeys(vocab, count)
    # print(neg_dict)


def remove_punt(set):
    set_copy = {
    i.replace('!', '').replace(')', '').replace('(', '').replace('<br', '').replace('/>', '').replace('*', '').replace(
        '?', '').replace('.', '').replace('"', '').replace(':', '').replace(';', '').replace(',', '').replace('/','').replace(
        '[', '').replace('--','').replace('_','').replace('-',' ').replace('{','').replace('}','').replace('\'','')
    for i in set}


    return set_copy


def print_words():
    words = read_words(listing_neg)
    # print("\n", len(words)) # 176435
    # print(words)


def main():

    read_words(listing_neg)

    print_words()

    words = read_words(listing_neg)

    my_set = set(read_words(listing_neg))

    word_freq(my_set)


if __name__ == "__main__":
    main()

