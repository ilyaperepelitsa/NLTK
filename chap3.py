import nltk
from nltk import word_tokenize

s = "Some sentence with random stuff"
print(nltk.pos_tag(word_tokenize(s)))

for i, x in nltk.pos_tag(word_tokenize(s)):
    if x in ["NN", "NNP"]:
        print(i)

# from nltk.tag.stanford import StanfordTagger
from nltk.tag.stanford import StanfordPOSTagger
from nltk.tag import stanford
st = StanfordPOSTagger("/Users/ilyaperepelitsa/quant/NLTK/stanford/models/english-bidirectional-distsim.tagger", "/Users/ilyaperepelitsa/quant/NLTK/stanford/stanford-postagger.jar")

tokens = nltk.word_tokenize(s)
st.tag(tokens)







from nltk.corpus import brown
import nltk


tags = []
for word, tag in brown.tagged_words(categories = "news"):
    tags.append(word)

nltk.FreqDist(tags).most_common()




brown_tagged_sents = brown.tagged_sents(categories = "news")
default_tagger = nltk.DefaultTagger("NN")
print(default_tagger.evaluate(brown_tagged_sents))



##### NGRAMS

from nltk.tag import UnigramTagger
from nltk.tag import DefaultTagger
from nltk.tag import BigramTagger
from nltk.tag import TrigramTagger

train_data = brown_tagged_sents[:int(len(brown_tagged_sents) * 0.9)]
test_data = brown_tagged_sents[int(len(brown_tagged_sents) * 0.9) :]

unigram_tagger = UnigramTagger(train_data, backoff = default_tagger)
print(unigram_tagger.evaluate(test_data))


bigram_tagger = BigramTagger(train_data, backoff = unigram_tagger)
print(bigram_tagger.evaluate(test_data))

trigram_tagger = TrigramTagger(train_data, backoff = bigram_tagger)
print(trigram_tagger.evaluate(test_data))





### NE CHUNKING

import nltk
from nltk import ne_chunk
sent = "Mark is studying at Stanford University in California"
print(ne_chunk(nltk.pos_tag(word_tokenize(sent)), binary = False))
(S
    (PERSON Mark/NNP)
    is/VBZ
    studying/VBG
    at/IN
    (ORGANIZATION Stanford/NNP University/NNP)
    in/IN
    NY (GPE California/NNP)))




from nltk.tag.stanford import StanfordNERTagger
st = StanfordNERTagger("/Users/ilyaperepelitsa/quant/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz", "/Users/ilyaperepelitsa/quant/stanford-ner/stanford-ner.jar")

st.tag("Rami Eid is studying at Stony Brook University in NY".split())
