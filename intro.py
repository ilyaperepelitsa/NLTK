import nltk
from urllib import request
import re
from bs4 import BeautifulSoup
import operator

# from pretty import clean_html




link = "http://youtube.com"
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
# clean = nltk.clean_html(html)


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

soup = BeautifulSoup(html, "html.parser")


some_text = BeautifulSoup.get_text(soup)

somet_text = some_text.encode('utf-8')
# tokens = re.split("\W+", some_text)
tokens = [tok for tok in some_text.split()]
print(tokens)

# Pew 1
print(tokens[0:100])

freq_dis = {}

for tok in tokens:
    if tok in freq_dis:
        freq_dis[tok] += 1
    else:
        freq_dis[tok] = 1

sorted_freq_dist = sorted(freq_dis.items(), key = operator.itemgetter(1), reverse = True)

print(sorted_freq_dist)

Freq_dist_nltk = nltk.FreqDist(tokens)
print(Freq_dist_nltk)
for k, v in Freq_dist_nltk.most_common():
    print(str(k) + " : " + str(v))


for k, v in Freq_dist_nltk.items():
    print(str(k) + " : " + str(v))

Freq_dist_nltk.plot(50, cumulative = False)

from nltk.corpus import stopwords
# nltk.download()
stop = set(stopwords.words('english'))

# stopwords = [word.strip().lower() for word in open("/usr/local/bin/python3/english.stop.txt")]
clean_tokens = [tok for tok in tokens if len(tok.lower()) > 1 and (tok.lower() not in stop)]

Freq_dist_nltk = nltk.FreqDist(clean_tokens)
Freq_dist_nltk.plot(20, cumulative = False)



def simple_parser(some_link, number_of_words):

    response = request.urlopen(some_link)
    html = response.read()

    soup = BeautifulSoup(html, "html.parser")
    some_text = BeautifulSoup.get_text(soup)
    somet_text = some_text.encode('utf-8')
    tokens = [tok for tok in some_text.split()]

    Freq_dist_nltk = nltk.FreqDist(tokens)


    from nltk.corpus import stopwords
    stop = set(stopwords.words('english'))
    clean_tokens = [tok for tok in tokens if len(tok.lower()) > 1 and (tok.lower() not in stop)]

    Freq_dist_nltk = nltk.FreqDist(clean_tokens)
    # return(Freq_dist_nltk.plot(number_of_words, cumulative = False))
    for k, v in Freq_dist_nltk.most_common():
        print(str(k) + " : " + str(v))



simple_parser("http://www.pornhub.com", 30)
