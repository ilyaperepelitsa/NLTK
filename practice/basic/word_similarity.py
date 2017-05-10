from nltk.corpus import wordnet

# wu-palmer similarity
cb = wordnet.synset("cookbook.n.01")
ib = wordnet.synset("instruction_book.n.01")

cb.wup_similarity(ib)

ref = cb.hypernyms()[0]
cb.shortest_path_distance(ref)
ib.shortest_path_distance(ref)
cb.shortest_path_distance(ib)



dog = wordnet.synsets("dog")[0]
dog.wup_similarity(cb)

## what does dog share with cookbook

sorted(dog.common_hypernyms(cb))


cook = wordnet.synset("cook.v.01")
bake = wordnet.synset("bake.v.02")

cook.wup_similarity(bake)


## path and laeacock chordorow similarity (LCH)
cb.path_similarity(ib)
cb.path_similarity(dog)
cb.lch_similarity(ib)
cb.lch_similarity(dog)
