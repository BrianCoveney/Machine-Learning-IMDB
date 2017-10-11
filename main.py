import os
import re

path_neg = '/home/brian/Desktop/4thYr_MachineLearning/Assignment/LargeIMDB/neg'
listing_neg = os.listdir(path_neg)

path_pos = '/home/brian/Desktop/4thYr_MachineLearning/Assignment/LargeIMDB/neg'
listing_pos = os.listdir(path_pos)


def readWords(list_files, path):

    # Initialize an empty Set
    vocab = set([])

    for eachFile in list_files:
        f = open(path + "/" + eachFile, "r", encoding='utf8')

        # Clean the dataset
        words = removeDigits(f)
        words_copy = [x.lower() for x in words]

        # Update the set with the dataset
        vocab.update(words_copy)
        f.close()

    # Clean the set of punctuation, etc
    vocab_copy = removePunt(vocab)

    # print(vocab_copy)
    return vocab_copy


def wordFreq(words_cleaned_set):
    frequency = {}
    for w in words_cleaned_set:
        match_pattern = re.findall(r'\b[a-z]{3,15}\b', w)

        for word in match_pattern:
            count = frequency.get(word, 0)
            frequency[word] = count + 1

        frequency_list = frequency.keys()

    word_feq_set = []
    for words in frequency_list:
        wd_map = words, frequency[words]
        word_feq_set.append(wd_map)

    return word_feq_set


def removeDigits(file):
    return file.read().replace('0', '').replace('1', '').replace('2', '').replace('3', '').replace('4', '') \
        .replace('5', '').replace('6', '').replace('7', '').replace('8', '').replace('9', '').split()


def removePunt(set):
    set_copy = {
        i.replace('!', '').replace(')', '').replace('(', '').replace('<br', '').replace('/>', '').replace('*',
                                                                                                          '').replace(
            '?', '').replace('.', '').replace('"', '').replace(':', '').replace(';', '').replace(',', '').replace('/',
                                                                                                                  '').replace(
            '[', '').replace('--', '').replace('_', '').replace('-', ' ').replace('{', '').replace('}', '').replace(
            '\'', '')
        for i in set}

    return set_copy


def main():
    # Negative reviews
    neg_words_cleaned_set = readWords(listing_neg, path_neg)
    neg_words_cleaned_set_copy = set(neg_words_cleaned_set)
    neg_word_freq_set = wordFreq(neg_words_cleaned_set_copy)
    print(neg_word_freq_set)

    # Positive reviews
    # pos_words_cleaned_set = readWords(listing_pos, path_pos)
    # pos_words_cleaned_set_copy = set(pos_words_cleaned_set)
    # pos_word_freq_set = wordFreq(pos_words_cleaned_set_copy)
    # print(pos_word_freq_set)


if __name__ == "__main__":
    main()

