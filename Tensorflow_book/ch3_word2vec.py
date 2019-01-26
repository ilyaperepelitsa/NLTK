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
# dictionary
# count[0:30]
data_index = 0


def generate_batch_skip_gram(batch_size, window_size):
    # data_index is updated by 1 everytime we read a data point
    global data_index

    # two numpy arras to hold target words (batch)
    # and context words (labels)
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    # span defines the total window size, where
    # data we consider at an instance looks as follows.
    # [ skip_window target skip_window ]
    span = 2 * window_size + 1

    # The buffer holds the data contained within the span
    buffer = collections.deque(maxlen=span)

    # Fill the buffer and update the data_index
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    # This is the number of context words we sample for a single target word
    num_samples = 2*window_size

    # We break the batch reading into two for loops
    # The inner for loop fills in the batch and labels with
    # num_samples data points using data contained withing the span
    # The outper for loop repeat this for batch_size//num_samples times
    # to produce a full batch
    for i in range(batch_size // num_samples):
        k=0
        # avoid the target word itself as a prediction
        # fill in batch and label numpy arrays
        for j in list(range(window_size))+list(range(window_size+1,2*window_size+1)):
            batch[i * num_samples + k] = buffer[window_size]
            labels[i * num_samples + k, 0] = buffer[j]
            k += 1

        # Everytime we read num_samples data points,
        # we have created the maximum number of datapoints possible
        # withing a single span, so we need to move the span by 1
        # to create a fresh new span
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

print('data:', [reverse_dictionary[di] for di in data[:8]])

for window_size in [1, 2]:
    data_index = 0
    batch, labels = generate_batch_skip_gram(batch_size=8, window_size=window_size)
    print('\nwith window_size = %d:' %window_size)
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])


batch_size = 128 # Data points in a single batch
embedding_size = 128 # Dimension of the embedding vector.
window_size = 4 # How many words to consider left and right.

# We pick a random validation set to sample nearest neighbors
valid_size = 16 # Random set of words to evaluate similarity on.
# We sample valid datapoints randomly from a large window without always being deterministic
valid_window = 50

# When selecting valid examples, we select some of the most frequent words as well as
# some moderately rare words as well
valid_examples = np.array(random.sample(range(valid_window), valid_size))
# valid_examples
valid_examples = np.append(valid_examples,random.sample(range(1000, 1000+valid_window), valid_size),axis=0)
# valid_examples
num_sampled = 32 # Number of negative examples to sample.



tf.reset_default_graph()

# Training input data (target word IDs).
train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
# Training input label data (context word IDs)
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
# Validation input data, we don't need a placeholder
# as we have already defined the IDs of the words selected
# as validation data
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
