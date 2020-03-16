from treccast import *
from sklearn.metrics.pairwise import cosine_similarity as cs
import re
import nltk
from spacy.lang.en import English

nlp = English()

paragraph_map = {}

has = dict()
has1 = {}

with open('data/evaluation/bert_uw_return.txt', 'r') as fp:
    line = fp.readline().split()
    rank = line[2].strip('[]',)
    print(rank)
