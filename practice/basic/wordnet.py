from nltk.corpus import wordnet
syn = wordnet.synsets("cookbook")[0]
syn.name()
syn.definition()

# a list of sunsets
[ word.definition() for word in wordnet.synsets("banana")]

wordnet.synset("cookbook.n.01")
wordnet.synsets('cooking')[0].examples()


syn.hypernyms()
syn.hypernyms()[0].hyponyms()
syn.root_hypernyms()
syn.hypernym_paths()
