para = "Hello World. It's good to see you. Thanks for buying this book."

from nltk.tokenize import sent_tokenize
sent_tokenize(para)


# to avoid on demand loading for better speed

import nltk.data
tokenizer = nltk.data.load("tokenizers/punkt/PY3/english.pickle")
tokenizer.tokenize(para)

from nltk.tokenize import word_tokenize
word_tokenize("Hello world.")


# to avoid loading on call
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
tokenizer.tokenize("Hello world.")

# word_tokenize("can't")



# keeps punctuation with the word
# doesn't seem to work
# from nltk.tokenize import PunktWordTokenizer


from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()
tokenizer.tokenize("Can't is a contraction")


## REGEX tokenizer

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer("[\w']+")
tokenizer.tokenize("Can't is a contraction.")

# if don't wanna instantiate the class
from nltk.tokenize import regexp_tokenize
regexp_tokenize("Can't is a contraction", "[\w']+")

# tokenize on white space

tokenizer = RegexpTokenizer('\s+', gaps = True)
tokenizer.tokenize("Can't is a contraction.")



from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import webtext
text = webtext.raw("overheard.txt")
sent_tokenizer = PunktSentenceTokenizer(text)


sents1 = sent_tokenizer.tokenize(text)
sents1[0]
from nltk.tokenize import sent_tokenize
sents2 = sent_tokenize(text)
sents2[0]


sents1[678]
sents2[678]


### GOOD idea to train your own tokenizers when the structure is "special"


with open("/usr/share/nltk_data/corpora/webtext/overheard.txt", encoding = "ISO-8859-2") as f:
    text = f.read()
