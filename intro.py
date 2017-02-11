import nltk
from urllib import request
import re
from bs4 import BeautifulSoup
import operator

from pretty import clean_html



link = "http://python.org"
response = request.urlopen(link)
html = response.read()


########1
# print(len(html))
#
# tokens = [tok for tok in html.split()]
# print("Total number of tokens : " + str(len(tokens)))
#
# print(tokens[0:20])

######2

# tokens = re.split("\W+", html)
# print(len(tokens))
# print(tokens[0:10])



#### 3
# soup = BeautifulSoup(html, "lxml")
# html = soup.get_text()
#
# tokens = [tok.encode('utf-8') for tok in html.split()]
# print(tokens[:100])



##### 3.5
clean = pretty.clean_html(html)


#####4

# freq_dis = {}
# for tok in tokens:
#     if tok in freq_dis:
#         freq_dis[tok] += 1
#     else: freq_dis[tok] = 1
#
# sorted_freq_dist = sorted(freq_dis.items(),
#                             key = operator.itemgetter(1),
#                             reverse = True)
#
# print(sorted_freq_dist[:25])


######5

# Freq_dist_nltk = nltk.FreqDist(tokens)
#
# for k, v in Freq_dist_nltk.items():
#     print(str(k) + " : " + str(v))
#
# Freq_dist_nltk.plot(50, cumulative = False)
