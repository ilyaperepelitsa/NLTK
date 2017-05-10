from nltk.corpus import wordnet
syn = wordnet.synsets("cookbook")[0]

syn.pos()


len(wordnet.synsets("great"))
len(wordnet.synsets("great", pos = "n"))
len(wordnet.synsets("great", pos = "a"))
