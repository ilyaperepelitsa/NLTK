from nltk.corpus import webtext
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

words = [w.lower() for w in webtext.words("grail.txt")]

# words

bcf = BigramCollocationFinder.from_words(words)
bcf.nbest(BigramAssocMeasures.likelihood_ratio, 30)


## remove punctuation and stopwords

from nltk.corpus import stopwords
stopset = set(stopwords.words("english"))

filter_stops = lambda w: len(w) < 3 or w in stopset
filter_stops
bcf.apply_word_filter(filter_stops)
bcf.nbest(BigramAssocMeasures.likelihood_ratio, 30)


### trigrams work the same way

from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import TrigramAssocMeasures

words = [w.lower() for w in webtext.words("singles.txt")]

tcf = TrigramCollocationFinder.from_words(words)

tcf.apply_word_filter(filter_stops)
tcf.apply_freq_filter(2)
tcf.nbest(TrigramAssocMeasures.likelihood_ratio, 10)

## consult NLTK API - NgramAssocMeasures in nltk.metrics package
