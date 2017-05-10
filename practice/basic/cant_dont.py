
import re


replacement_patterns = [
    ("won't", "will not"),
    ("can't", "cannot"),
    ("i'm", "i am"),
    ("ain't", "is not"),
    ("(\w+)'ll", "\g<1> will"),
    ("(\w+)n't", "\g<1> not"),
    ("(\w+)'ve", "\g<1> have"),
    ("(\w+)'s", "\g<1> is"),
    ("(\w+)'re", "\g<1> are"),
    ("(\w+)'d", "\g<1> would"),
    ("&", "and")]

class RegexpReplacer(object):
    def __init__(self, patterns = replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]

    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            s = re.sub(pattern, repl, s)
        return s


replacer = RegexpReplacer()
replacer.replace("can't is a contraction")
replacer.replace("I should've done that thing I didn't do")

# two tokenizers
from nltk.tokenize import word_tokenize

word_tokenize("can't is a contraction")
word_tokenize(replacer.replace("can't is a contraction"))
