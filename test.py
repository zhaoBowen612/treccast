from treccast import *
from sklearn.metrics.pairwise import cosine_similarity as cs
import re
import nltk
from spacy.lang.en import English

nlp = English()

paragraph_map = {}

has = {1: 2, 2: 3, 3: 4, 4: 5}

print(has[1])
print(has.get(5))
print(has[5])


# with open('data/evaluation/bert_uw_return.txt', 'r') as fp:
#     line = fp.readline().split()
#     rank = line[2].strip('[]',)
#     print(rank)
