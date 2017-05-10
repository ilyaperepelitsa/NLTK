from nltk.corpus import stopwords
english_stops = set(stopwords.words("english"))

words = ["Cant't", 'is', 'a', 'contraction']
[word for word in words if word not in english_stops]

# list of all stop words
stopwords.fileids()
stopwords.words("russian")
