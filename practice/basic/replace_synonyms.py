class WordReplacer(object):
    def __init__(self, word_map):
        self.word_map = word_map

    def replace(self, word):
        return self.word_map.get(word, word)

def pew():
    print("pew")


replacer = WordReplacer({"bday" : "birthday"})
replacer.replace("bday")
replacer.replace("happy")

## using CSV

import csv

class CsvWordReplacer(WordReplacer):
    def __init__(self, fname):
        word_map = {}
        for line in csv.reader(open(fname)):
            word, syn = line
            word_map[word] = syn
        super(CsvWordReplacer, self).__init__(word_map)

### uncomment - import stuff
# replacer = CsvWordReplacer("synonyms.csv")
# replacer.replace("bday")

## using PyYAML

import yaml

class YamlWordReplacer(WordReplacer):
    def __init__(self, name):
        word_map = yaml.load(open(fname))
        super(YamlWordReplacer, self).__init__(word_map)

### uncomment - import stuff
# replacer = YamlWordReplacer("synonyms.yaml")
