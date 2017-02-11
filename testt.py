# import re
# import nltk
import sys

# mystring = "Monty Python ! And the holy grail ! \n the the a monty"

# No argument - treat space as delimiter
# print(mystring.split())
#
# print(mystring.strip())
# print(mystring.upper())
# print(mystring.lower())
#
# if re.search("Python", mystring):
#     print("FOund SOmething")
# else:
#     print("Nah")
#
#
# print(re.findall("!", mystring))

# word_freq = {}
# for tok in mystring.split():
#     if tok in word_freq:
#         word_freq[tok] += 1
#     else:
#         word_freq[tok] = 1
#
# print(word_freq)

def word_freq(mystring):
    """
    Function to generate freq distr of the given text
    """
    print(mystring)
    word_freq = {}
    for tok in mystring.split():
        if tok in word_freq:
            word_freq[tok] += 1
        else:
            word_freq[tok] = 1

    print(word_freq)


def main():
    str = "this is my first function"
    word_freq(str)
if __name__ == "__main__":
    main()
