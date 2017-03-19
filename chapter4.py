# toy CFG (Context Free Grammar)
import nltk
from nltk import CFG
toy_grammar = CFG.fromstring('''/
S -> NP VP
VP -> V NP
V -> "eats" | "drinks"
NP -> Det N
Det -> "a" | "an" | "the"
N -> "president" | "Obama" | "apple" | "coke"
''')

import sys
# prints whether python is version 3 or not
print(sys.version_info.major)
toy_grammar = CFG.fromstring("""
S -> NP VP                 # S indicate the entire sentence
VP -> V NP                  # VP is verb phrase the
V -> "eats" | "drinks"     # V is verb
NP -> Det N                 # NP is noun phrase (chunk that has noun in it)
Det -> "a" | "an" | "the"    # Det is determined used in the sentences
N -> "president" | "Obama" | "apple" | "coke" # N some example nouns
""")


grammar = CFG.fromstring("""S -> NP VP, PP -> P NP, NP -> Det N | NP PP, VP -> V NP | VP PP, Det -> 'a' | 'the', N -> 'dog' | 'cat', V -> 'chased' | 'sat', P -> 'on' | 'in' """)


from nltk.chunk import *
# chunk_rules = nltk.ChunkRule("<.*>+", "chunk everything")
from nltk.chunk.regexp import *
reg_parser = RegexpParser('''
    NP: {<DT>? <JJ>* <NN>*}     # NP
     P: {<IN>}                  # Preposition
     V: {<V.*>}                 # Verb
    PP: {<P> <NP>}              # PP -> P NP
    VP: {<V> <NP|PP>*}          # VP -> V (NP|PP)* ''')

test_sent = "Mr. Obmama played a big role in the Health insurance bill"
test_sent_pos = nltk.pos_tag(nltk.word_tokenize(test_sent))
paresed_out = reg_parser.parse(test_sent_pos)
print(paresed_out)




from nltk.chunk.regexp import *
test_sent = '''\
The prime minister announced he had asked the chief\
 government whip, Phillip Ruddock, to call a special party room meeting\
 for 9am on Monday to consider the spill motion.'''
test_sent
test_sent_pos = nltk.pos_tag(nltk.word_tokenize(test_sent))
test_sent_pos


rule_vp = ChunkRule("(<VB.*>)?(<VB.*>)+(<PRP>)?", "Chunk VPs")
parser_vp = RegexpChunkParser([rule_vp], chunk_label = "VP")
print(parser_vp.parse(test_sent_pos))


rule_np = ChunkRule("(<DT>?<RB>?)?<JJ|CD>*(<JJ|CD><,>)*(<NN.*>)+", 'Chunk NPs')
parser_np = RegexpChunkParser([rule_np], chunk_label = "NP")
print(parser_np.parse(test_sent_pos))





### NAMED ENTITY RECOGNITION
import nltk
from urllib import request
import re
from bs4 import BeautifulSoup

def simple_parser(some_link):
    response = request.urlopen(some_link)
    html = response.read()

    soup = BeautifulSoup(html, "html.parser")
    some_text = BeautifulSoup.get_text(soup)
    some_text = some_text.encode('utf-8')
    return str(some_text)


newtext = simple_parser("http://food.com")

sentences  = str(nltk.sent_tokenize(newtext))
# sentences
tokenized = [nltk.word_tokenize(sentences) for sentence in sentences]
