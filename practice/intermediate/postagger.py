from nltk.corpus.reader import TaggedCorpusReader
reader = TaggedCorpusReader(".", ".*\.pos")
reader.words()
reader.tagged_words()
reader.sents()
reader.tagged_sents()
reader.paras()
reader.tagged_paras()



### customize word tokenizer
from nltk.tokenize import SpaceTokenizer
reader = TaggedCorpusReader(".", ".*\.pos", word_tokenizer = SpaceTokenizer())
reader.words()


### customize line tokenizer
from nltk.tokenize import LineTokenizer
reader = TaggedCorpusReader(".", ".*\.pos", sent_tokenizer = LineTokenizer())
reader.sents()


## mapping to universal TaggedCorpusReader
reader = TaggedCorpusReader(".", ".*\.pos", tagset = "en-brown")
reader.tagged_words(tagset = "universal")

from nltk.corpus import treebank
treebank.tagged_words()
treebank.tagged_words(tagset = "universal")
treebank.tagged_words(tagset = "brown")
