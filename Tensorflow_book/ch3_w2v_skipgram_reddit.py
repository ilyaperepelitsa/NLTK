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
working_params = hyper_params["ch3_w2v_skipgram_reddit"]


DATA = pd.read_csv("/Users/ilyaperepelitsa/quant/NLTK/data/reddit-comment-score-prediction/comments_negative.csv")
DATA_2 = pd.read_csv("/Users/ilyaperepelitsa/quant/NLTK/data/reddit-comment-score-prediction/comments_positive.csv")

DATA = pd.concat([DATA, DATA_2], axis = 0)

del(DATA_2)

DATA.head()

DATA = pd.concat([DATA['text'], DATA["parent_text"]], axis = 0)
DATA
text = DATA.str.cat(sep = ". ")
# len(text)
# text

lower_text = text.lower()
tokenized_text = nltk.word_tokenize(lower_text)
# len(tokenized_text)/1000000

VOCABULARY_SIZE = working_params["vocabulary_size"]
NUM_STEPS = working_params["num_steps"]
BATCH_SIZE = working_params["minibatch_size"]
EMBEDDING_SIZE = working_params["embedding_size"]
WINDOW_SIZE = working_params["window_size"]
VALID_SIZE = working_params["valid_size"]
VALID_WINDOW = working_params["valid_window"]
NUM_SAMPLED = working_params["num_sampled"]
PRINT_AND_SAVE = working_params["print_and_save"]


def build_dataset(words, vocabulary_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
          index = dictionary[word]
        else:
          index = 0  # dictionary['UNK']
          unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    assert len(dictionary) == vocabulary_size
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words = tokenized_text,
                                            vocabulary_size = VOCABULARY_SIZE)



# sorted(data, reverse=True)
# count
# sortdictionary.items()
# len(data)
# len(count)
# len(dictionary)
# dictionary.values()
# count[0:100]

data_index = 0
data_index
# generate_batch_skip_gram(128, 5)

# generate_batch_skip_gram(64, 3)
# len(data)

def generate_batch_skip_gram(batch_size, window_size):
    global data_index
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * window_size + 1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    num_samples = 2*window_size
    for i in range(batch_size // num_samples):
        k=0
        for j in list(range(window_size))+list(range(window_size+1,2*window_size+1)):

            batch[i * num_samples + k] = buffer[window_size]
            labels[i * num_samples + k, 0] = buffer[j]
            k += 1
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels
    # return batch

# 64 // 4

# generate_batch_skip_gram(120, 3)
#
# pd.Series(pew)

# generate_batch_skip_gram(64, 6)
# 128/6

# data[537537629]
# print('data:', [reverse_dictionary[di] for di in data[:8]])
# for window_size in [15, 2]:
#     data_index = 0
#     batch, labels = generate_batch_skip_gram(batch_size=8, window_size=window_size)
#     print('\nwith window_size = %d:' %window_size)
#     print('    batch:', [reverse_dictionary[bi] for bi in batch])
#     print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])
# generate_batch_skip_gram(BATCH_SIZE, WINDOW_SIZE)


valid_examples = np.array(random.sample(range(VALID_WINDOW), VALID_SIZE))

valid_examples = np.append(valid_examples,
                random.sample(range(1000, 1000+VALID_WINDOW), VALID_SIZE),axis=0)

g = tf.Graph()


# dictionary.keys()


with g.as_default():

    train_dataset = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    train_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1])

    with tf.name_scope("validation_dataset") as scope:
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.name_scope("embeddings") as scope:
        embeddings = tf.Variable(tf.random_uniform([VOCABULARY_SIZE,
                                                EMBEDDING_SIZE], -1.0, 1.0))

    with tf.name_scope("softmax") as scope:
        softmax_weights = tf.Variable(
            tf.truncated_normal([VOCABULARY_SIZE, EMBEDDING_SIZE],
                                stddev=0.5 / math.sqrt(EMBEDDING_SIZE))
        )
        softmax_biases = tf.Variable(tf.random_uniform([VOCABULARY_SIZE],0.0,0.01))

    with tf.name_scope("embed_lookup") as scope:
        embed = tf.nn.embedding_lookup(embeddings, train_dataset)

    with tf.name_scope("loss") as scope:
        loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(
                weights=softmax_weights, biases=softmax_biases, inputs=embed,
                labels=train_labels, num_sampled=NUM_SAMPLED, num_classes=VOCABULARY_SIZE)
        )

    with tf.name_scope("normalization") as scope:
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm

    with tf.name_scope("validation_embeddings") as scope:
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)

    with tf.name_scope("similarity") as scope:
        similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))


    # Optimizer.
    with tf.name_scope("train") as scope:
        optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
# optimizer

    with tf.Session() as session:
        saver = tf.train.Saver()
        # saver = tf.train.Saver({'embeddings':embeddings})

        # num_steps = 300001
        skip_losses = []
        average_loss = 0
        # learning_counter = 0
        # prev_loss_test = 0
        # current_loss_test = 0
        try:
            saver.restore(session, os.path.join(LOGS_PATH, working_params["logs"], "model.ckpt"))
        except (tf.errors.InvalidArgumentError):
        # session.run(init)
            session.run(tf.global_variables_initializer())

        for step in range(NUM_STEPS):
            batch_data, batch_labels = generate_batch_skip_gram(
                                            BATCH_SIZE, WINDOW_SIZE)
            # print(batch_data)
            # print(batch_labels)

            feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += l

            if (step+1) % int(PRINT_AND_SAVE / 5) == 0:
                # print(step)
                if step > 0:
                    average_loss = average_loss / PRINT_AND_SAVE / 5

                skip_losses.append(average_loss)
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step %d: %f' % (step+1, average_loss))
                average_loss = 0

                if (step+1) % (PRINT_AND_SAVE * 10) == 0:
                    sim = similarity.eval()
                # Here we compute the top_k closest words for a given validation word
                # in terms of the cosine distance
                # We do this for all the words in the validation set
                # Note: This is an expensive step
                    for i in range(VALID_SIZE):
                        valid_word = reverse_dictionary[valid_examples[i]]
                        top_k = 8 # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k+1]
                        log = 'Nearest to %s:' % valid_word
                        for k in range(top_k):
                            close_word = reverse_dictionary[nearest[k]]
                            log = '%s %s,' % (log, close_word)
                        print(log)
                        # if step % working_params["print_and_save"] == 0:

                    saver.save(session,
                            os.path.join(LOGS_PATH,
                                            working_params["logs"],
                                            "model.ckpt"))
        skip_gram_final_embeddings = normalized_embeddings.eval()



# tensorboard --logdir=./logs/basic_word2vec_ch3_scienfeld/ --host 127.0.0.1


# import zipfile
# from tensorflow.contrib.tensorboard.plugins import projector
# import csv
#
# saver2 = tf.train.Saver({'embeddings':embeddings})
# session = tf.InteractiveSession()
# saver2.save(session,
#         os.path.join(LOGS_PATH,
#                         working_params["logs"],
#                         "model_2.ckpt"), 0)

# with open(os.path.join(LOGS_PATH,
#                 working_params["logs"], 'metadata.tsv'), 'w',encoding='utf-8') as csvfile:
#     writer = csv.writer(csvfile, delimiter='\t',
#                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     writer.writerow(['Word','Word ID'])
#     for w, wi in dictionary.items():
#       writer.writerow([w,wi])

# list(dictionary.keys())[0:5]

words = '\n'.join(list(dictionary.keys()))
with open(os.path.join(LOGS_PATH,
                working_params["logs"], 'metadata.tsv'), 'w',encoding='utf-8') as f:
   f.write(words)



config = projector.ProjectorConfig()

# You can add multiple embeddings. Here we add only one.
embedding_config = config.embeddings.add()
embedding_config.tensor_name = embeddings.name
# Link this tensor to its metadata file (e.g. labels).
embedding_config.metadata_path = os.path.join(LOGS_PATH,
                                working_params["logs"], 'metadata.tsv')

# Use the same LOG_DIR where you stored your checkpoint.
summary_writer = tf.summary.FileWriter(os.path.join(LOGS_PATH,
                                            working_params["logs"]))

# The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
# read this file during startup.
projector.visualize_embeddings(summary_writer, config)
# tensorboard --logdir=./logs/basic_word2vec_ch3_v3/ --host 127.0.0.1
