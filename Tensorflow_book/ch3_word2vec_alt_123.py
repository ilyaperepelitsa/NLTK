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
# nltk.download()
import json
import pandas as pd


ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath("__file__")), "Tensorflow_book")
JSON_PATH = os.path.join(os.path.dirname(os.path.abspath("__file__")),"Tensorflow_book", "tf_specs.json")
LOGS_PATH = os.path.join(ROOT_PATH, "logs")
SUMMARY_PATH = os.path.join(ROOT_PATH, "summary")
METRICS_PATH = os.path.join(ROOT_PATH, "metrics.json")



def check_dir_create(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

check_dir_create(LOGS_PATH)
hyper_params = json.load(open(JSON_PATH))
working_params = hyper_params["basic_word2vec_ch3_v1"]


DATA = pd.read_csv("/Volumes/data_pew/text_data/stack_overflow_pandas/SO_pandas.csv")
# pew
text = DATA['Markdown'].str.cat(sep = ". ")
# len(text)
lower_text = text.lower()
tokenized_text = nltk.word_tokenize(lower_text)
# len(tokenized_text)/1000000



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
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
# del words  # Hint to reduce memory.
# data


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
valid_examples = np.append(valid_examples,random.sample(range(1000, 1000+valid_window), valid_size),axis=0)

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


# Embedding layer, contains the word embeddings
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

# Softmax Weights and Biases
softmax_weights = tf.Variable(
    tf.truncated_normal([vocabulary_size, embedding_size],
                        stddev=0.5 / math.sqrt(embedding_size))
)
softmax_biases = tf.Variable(tf.random_uniform([vocabulary_size],0.0,0.01))


embed = tf.nn.embedding_lookup(embeddings, train_dataset)

# Compute the softmax loss, using a sample of the negative labels each time.
loss = tf.reduce_mean(
    tf.nn.sampled_softmax_loss(
        weights=softmax_weights, biases=softmax_biases, inputs=embed,
        labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size)
)


norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)


num_steps = 100001
skip_losses = []
# ConfigProto is a way of providing various configuration settings
# required to execute the graph

generate_batch_skip_gram(batch_size, window_size)


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
  # Initialize the variables in the graph
  tf.global_variables_initializer().run()
  print('Initialized')
  average_loss = 0

  # Train the Word2vec model for num_step iterations
  for step in range(num_steps):

    # Generate a single batch of data
    batch_data, batch_labels = generate_batch_skip_gram(
      batch_size, window_size)

    # Populate the feed_dict and run the optimizer (minimize loss)
    # and compute the loss
    feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
    _, l = session.run([optimizer, loss], feed_dict=feed_dict)

    # Update the average loss variable
    average_loss += l

    if (step+1) % 2000 == 0:
      if step > 0:
        average_loss = average_loss / 2000

      skip_losses.append(average_loss)
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step %d: %f' % (step+1, average_loss))
      average_loss = 0

    # Evaluating validation set word similarities
    if (step+1) % 10000 == 0:
      sim = similarity.eval()
      # Here we compute the top_k closest words for a given validation word
      # in terms of the cosine distance
      # We do this for all the words in the validation set
      # Note: This is an expensive step
      for i in range(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log = 'Nearest to %s:' % valid_word
        for k in range(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log = '%s %s,' % (log, close_word)
        print(log)
  skip_gram_final_embeddings = normalized_embeddings.eval()



generate_batch_skip_gram(batch_size, window_size)
