from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stemmer.stem("cooking")
stemmer.stem("cookery")


## lancaster class - more aggressive than porter stemmer

from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()
stemmer.stem("cooking")
stemmer.stem("cookery")


### regexp

# takes a single regexp and removes any suffix or prefix that matches it

from nltk.stem import RegexpStemmer
stemmer = RegexpStemmer("ing")
stemmer.stem("cooking")
stemmer.stem("cookery")
stemmer.stem("fucking")
stemmer.stem("ingleside")


## snowball stemmer - uses a bunch of languages

from nltk.stem import SnowballStemmer
SnowballStemmer.languages[0]
russian_stemmer = SnowballStemmer("russian")
russian_stemmer.stem("существовать")
