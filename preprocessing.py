import os
import nltk
import json
import random
import pickle
import zipfile
from collections import defaultdict

if __name__ == "__main__":
    print("Unzip text8.zip")
    unzip_file = zipfile.ZipFile(os.getcwd() + "/data/text8.zip")
    unzip_file.extract('text8',os.getcwd() + "/data/")
    unzip_file.close()
    print("Finish unzip text8.zip")

    print("Read text8")
    data = open(os.getcwd() + "/data/text8", 'r').readlines()
    tokenized_data = nltk.tokenize.word_tokenize(data[0])

    voca = sorted(list(set(tokenized_data)))
    voca_count = defaultdict(lambda: 0)
    for word in tokenized_data:
        voca_count[word] += 1
    print("Finish making voca_count")

    with open(os.getcwd() + "/data/tokenized_data.pickle", 'wb') as tokenized_data_pickle:
        pickle.dump(tokenized_data, tokenized_data_pickle)
    print("Save tokenized_data.pickle")
    json.dump(voca_count, open(os.getcwd() + "/data/voca_count", 'w'))
    print("Save voca_count")

