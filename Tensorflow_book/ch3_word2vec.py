import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import bz2
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import nltk # standard preprocessing
import operator # sorting items in dictionary by value
#nltk.download() #tokenizers/punkt/PY3/english.pickle
from math import ceil
import csv


import pandas as pd


pew = pd.read_csv("/Volumes/data_pew/text_data/stack_overflow_pandas/SO_pandas.csv")
pew
text = pew['Markdown'].str.cat(sep = ". ")
len(text)
lower_text = text.lower()
tokenized_text = nltk.word_tokenize(lower_text)


vocabulary_size = 50000

def build_dataset(words):
    count = [['UNK', -1]]
    # Gets only the vocabulary_size most common words as the vocabulary
    # All the other words will be replaced with UNK token
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()

    # Create an ID for each word by giving the current length of the dictionary
    # And adding that item to the dictionary
    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0
    # Traverse through all the text we have and produce a list
    # where each element corresponds to the ID of the word found at that index
    for word in words:
        # If word is in the dictionary use the word ID,
        # else use the ID of the special token "UNK"
        if word in dictionary:
          index = dictionary[word]
        else:
          index = 0  # dictionary['UNK']
          unk_count = unk_count + 1
        data.append(index)

    # update the count variable with the number of UNK occurences
    count[0][1] = unk_count

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    # Make sure the dictionary is of size of the vocabulary
    assert len(dictionary) == vocabulary_size

    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(tokenized_text)
dictionary[0:4]
