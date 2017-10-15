import os
import re

# path_pos = '/home/brian/Desktop/4thYr_MachineLearning/Assignment/LargeIMDB/pos'
# path_neg = '/home/brian/Desktop/4thYr_MachineLearning/Assignment/LargeIMDB/neg'
# listing_pos = os.listdir(path_pos)
# listing_neg = os.listdir(path_neg)

example_pos_path = 'res/pos'
example_pos_listings = os.listdir((example_pos_path))


def store_unique_words(listing, path):
    word_dict ={}

    for eachFile in listing:
        f = open(path + "/" + eachFile, "r", encoding='utf8')
        for line in f:
            for word in re.findall(r'[\w]+', line.lower()):
                word_dict[word] = word_dict.setdefault(word, 0) + 1

        # print each file, word and word freq -> file1 'that': 5 | file2 'that': 9 | file3 'that': 20
        # print(f)
        # print(word_dict)

    # print unique words and freq -> 'that': 20
    print(word_dict)

    return word_dict


def main():

    # Positive review words
    # store_unique_words(listing_pos, path_pos)

    # Negative review words
    # store_unique_words(listing_neg, path_neg)

    store_unique_words(example_pos_listings, example_pos_path)


if __name__ == "__main__":
    main()
