from nltk.corpus.reader import ChunkedCorpusReader
reader = ChunkedCorpusReader(".", ".*\.chunk")
reader.chunked_words()
reader.chunked_sents()
reader.chunked_paras()
# reader.words()
# reader.sents()
reader.chunked_sents()[1].draw()
