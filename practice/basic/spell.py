import enchant
from nltk.metrics import edit_distance

class SpellingReplacer(object):
    def __init__(self, dict_name = "en", max_dist = 2):
        self.spell_dict = enchant.Dict(dict_name)
        self.max_dist = max_dist

    def replace(self, word):
        if self.spell_dict.check(word):
            return word
        suggestions = self.spell_dict.suggest(word)

        if suggestions and edit_distance(word, suggestions[0]) <= self.max_dist:
            return suggestions[0]
        else:
            return word

replacer = SpellingReplacer(max_dist = 4)
replacer.replace("cookbok")
replacer.replace("langueeege")


## see edit distance

import enchant
d = enchant.Dict("en")
d.suggest("languege")


from nltk.metrics import edit_distance
edit_distance("language", "lenguage")
edit_distance("language", "languoasdwdwqds")


# what languages we have
enchant.list_languages()

# using other dictionaries

import enchant
dUS = enchant.Dict("en_US")
dUS.check("theater")

dGB = enchant.Dict("en_GB")
dGB.check("theater")

us_replacer = SpellingReplacer("en_US")
us_replacer.replace("theater")

gb_replacer = SpellingReplacer("en_GB")
gb_replacer.replace("theater")


# augment dictionary with personal word list
d = enchant.Dict("en_US")
d.check("nltk")

d = enchant.DictWithPWL("en_US", "mywords.txt")
d.check("nltk")


### create a custom class

class CustomSpellingReplacer(SpellingReplacer):
    def __init__(self, spell_dict, max_dist = 2):
        self.spell_dict = spell_dict
        self.max_dist = max_dist


d = enchant.DictWithPWL("en_US", "mywords.txt")
replacer = CustomSpellingReplacer(d)
replacer.replace("nltk")
