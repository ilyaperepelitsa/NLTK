import sys
import codecs
import os
f = "/Users/ilyaperepelitsa/Downloads/nytimes.txt"
# news_content = f.read()
path = "/Users/ilyaperepelitsa/Downloads/"
myfile = "nytimes.txt"
with open(os.path.join(path, myfile), "r") as infile:
    infile = codecs.open(os.path.join(path, myfile), encoding='utf-8')

f = infile.read()



https://наш.дом.рф/аналитика/застройщики/группа_компаний/755344001/регионы/таблица
