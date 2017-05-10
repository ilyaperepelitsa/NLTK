from nltk.corpus import wordnet
syn = wordnet.synsets("cookbook")[0]
lemmas = syn.lemmas()
len(lemmas)

[lemma.name() for lemma in lemmas]

lemmas[0].name()
lemmas[1].name()
lemmas[0].synset() == lemmas[1].synset()


# getting all the synonyms for the synset
[lemma.name() for lemma in syn.lemmas()]


synonyms = []
for syn in wordnet.synsets("book"):
    for lemma in syn.lemmas():
        synonyms.append(lemma.name())
len(synonyms)
len(set(synonyms))




### Antonyms

gn2 = wordnet.synset("good.n.02")
gn2.definition()

evil = gn2.lemmas()[0].antonyms()[0]
evil.name()

evil.synset().definition()


ga1 = wordnet.synset("good.a.01")
ga1.definition()
bad = ga1.lemmas()[0].antonyms()[0]
bad.name()
bad.synset().definition()
