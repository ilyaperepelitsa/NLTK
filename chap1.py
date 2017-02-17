import csv
import json

with open("/Users/ilyaperepelitsa/quant/rental/zillow_master1.csv", "rt") as f:
    reader = csv.reader(f, delimiter = ",", quotechar = '"')

    for line in reader:
        print(line[1])


jsonfile = open("/Users/ilyaperepelitsa/Downloads/rows.json")
data = json.load(jsonfile)
print(data)



inputstring = " THis is some sentence that the book didn't explicitly recommend to do? Pew pew pew!"
from nltk.tokenize import sent_tokenize


all_sent = sent_tokenize(inputstring)
print(all_sent)


###  Train our own sentence splitter

import nltk.tokenize.punkt
tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()


#### Tokenization and things

stringy = "Hello something somenthing7 shrewburry"

print(stringy.split())

from nltk.tokenize import word_tokenize

word_tokenize(stringy)


from nltk.tokenize import regexp_tokenize, wordpunct_tokenize, blankline_tokenize

regexp_tokenize(stringy, pattern = "\w+")
regexp_tokenize(stringy, pattern = "\d+")

wordpunct_tokenize(stringy)
blankline_tokenize(stringy)






#### Stemming

from nltk.stem import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer


pst = PorterStemmer()
lst = LancasterStemmer()


lst.stem("pornography")
pst.stem("pornography")


#### Lemmatization

from nltk.stem import WordNetLemmatizer

wlem = WordNetLemmatizer()
wlem.lemmatize("ate")



#### Stop word removal
from nltk.corpus import stopwords
stoplist = stopwords.words("english")

text = "This is ome text that I decided to type in order to test the thingy"
cleanwords = []
for i in text.split():
    if i not in stoplist:
        cleanwords.append(i)

print(cleanwords)


### Rare words removal    ------- DID NOT QUITE WORK

nltk.FreqDist(token)
print(nltk.FreqDist)


##### SPELL CORECTION

from nltk.metrics import edit_distance
edit_distance("rain", "shine")
