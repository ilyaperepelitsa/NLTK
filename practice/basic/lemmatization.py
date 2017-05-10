from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("cooking")
lemmatizer.lemmatize("cooking", pos = "v")
lemmatizer.lemmatize("cookbooks")


### comparing stemmer and lemmatizer

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stemmer.stem("believes")
lemmatizer.lemmatize("believes")
