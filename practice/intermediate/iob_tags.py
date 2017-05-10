from nltk.corpus.reader import ConllChunkCorpusReader
conllreader = ConllChunkCorpusReader(".", ".*\.iob", ("NP", "VP", "PP"))
conllreader.chunked_words()
conllreader.chunked_sents()
conllreader.iob_words()
conllreader.iob_sents()


### tree leaves

reader.chunked_words()[0].leaves()
reader.chunked_sents()[0].leaves()
reader.chunked_paras()[0][0].leaves()


### categorized text corpus

from nltk.corpus import brown
brown.categories()


### text categorization

from nltk.corpus.reader import CategorizedPlaintextCorpusReader
reader = CategorizedPlaintextCorpusReader(".", "movie_.*\.txt", cat_pattern = "movie_(\w+)\.txt")
reader.categories()
reader.fileids(categories = ["neg"])
reader.fileids(categories = ["pos"])

reader = CategorizedPlaintextCorpusReader(".", "movie_.*\.txt",
            cat_map = {"movie_pos.txt" : ["pos"], "movie_neg.txt" : ["neg"]})
reader.categories()
