import os, os.path

path = os.path.expanduser("~/nltk_data")

if not os.path.exists(path):
    os.mkdir(path)
# os.mkdir(path + "/corpora/cookbook")
os.path.exists(path)


import nltk.data
path in nltk.data.path

print(nltk.data.path)
import nltk.data
nltk.data.load("corpora/cookbook/mywords.txt", format = "raw")
nltk.data.load("corpora/cookbook/mywords.txt")



### corpus reader

from nltk.corpus.reader import WordListCorpusReader

reader = WordListCorpusReader("/Users/ilyaperepelitsa/nltk_data/corpora/cookbook/", ["wordlist.csv"])
reader.words()
reader.fileids()


reader.raw()
from nltk.tokenize import line_tokenize
line_tokenize(reader.raw())


### names wordlist corpus

from nltk.corpus import names
names.fileids()

len(names.words("female.txt"))
len(names.words("male.txt"))


### english word corpus

from nltk.corpus import words
words.fileids()

len(words.words("en-basic"))
# words.words("en-basic")
len(words.words("en"))
