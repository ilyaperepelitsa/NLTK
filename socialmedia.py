from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import sys

consumer_key = "zzjzyaKnfJkjPEw6ooXUJRBHS"
consumer_secret = "eJszE1XeNqL3kg0ZpmjSiufyQ3jXpGY9CXMzwh9v4ZhBiYVxYr"

access_token = "17262688-pBfACEdPssQFNgE0ndlxjBA69xJrrG3AGKHi2yKaq"
access_token_secret = "peerTYpIKMImpDtg048JnkLPkgsL3KfqYYgDYr1Taza08"



class StdOutListener(StreamListener):
      def on_data(self, data):
          with open(sys.argv[1], "a") as tf:
              tf.write(data)
          return

      def on_error(self, status):
          print(status)

if __name__ == "__main__":
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth,l)
    stream.filter(track = ["iphone"])


import json
import sys
# tweets = json.loads(open(sys.argv[1]).read())

tweets = []
tweet_texts = []
len(tweet_texts)

# for line in open(sys.argv[1], 'r'):
#     tweets.append(json.loads(line))
counter = 0
klout_scores = []

for line in open(sys.argv[1], 'r'):
    counter += 1
    tweet_obj = json.loads(line)
    keys = tweet_obj.keys()
    # print(keys)
    if "user" in tweet_obj:
        # tweet_texts.append(tweet_obj["text"])
        # print(str(tweet_obj["user"]["screen_name"]) + ": " + str(tweet_obj["user"]["followers_count"] / tweet_obj["user"]["friends_count"]))
        # print(str(tweet_obj["user"]["screen_name"]) + ": " + str(tweet_obj["user"]["followers_count"]))
        try:
            klout_scores.append(tweet_obj["user"]["screen_name"] + ": " + str(tweet_obj["user"]["followers_count"]/tweet_obj["user"]["friends_count"]))
        except ZeroDivisionError:
            continue
    else:
        print("______no text_____")
print(counter)

for i in klout_scores:
    print(i)

# tweet_texts = [tweet["text"] for tweet in tweets]
# tweet_source = [tweet["source"] for tweet in tweets]
# tweet_geo = [tweet["geo"] for tweet in tweets]
# tweet_locations = [tweet["place"] for tweet in tweets]
# hashtags = [hashtag["text"] for tweet in tweets for hashtag in tweet["entities"]["hashtags"]]

print(tweet_texts)
print(tweet_locations)
print(tweet_geo)
print(hashtags)




import nltk
from nltk import word_tokenize, sent_tokenize
from nltk import FreqDist
tweets_tokens = []

for tweet in tweet_texts:
    tweets_tokens.append(word_tokenize(tweet))
#
# Topic_distribution = nltk.FreqDist(tweets_tokens)
# Freq_dist_nltk.plot(50, cumulative = False)

import nltk
Topics = []
for tweet in tweet_texts:
    tagged = nltk.pos_tag(word_tokenize(tweet))
    Topics_token = [word for word, pos in tagged if pos in ["NN", "NNP"]]



for line in open(sys.argv[1], 'r'):
    counter += 1
    tweet_obj = json.loads(line)
    keys = tweet_obj.keys()
    # print(keys)
    if "text" in tweet_obj:
        tweet_texts.append(tweet_obj["text"])
    else:
        print("______no text_____")
