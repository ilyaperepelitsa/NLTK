import sys
import datetime
import pickle
import nltk
nltk.download("punkt")

for line in sys.stdin:
    line = line.strip()
    print(line)
    id, content = line.split("\t")
    print(tok.tokenize(content))
    tokens = nltk.word_tokenize(concat_all_text)
    print("\t".join([id, content, tokens]))


# root@localhost: j24mufr:KJxi
