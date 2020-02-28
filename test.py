from treccast import *
from spacy.lang.en import English
import nltk

paragraph_map = {}
with open('MARCO_1.txt', 'r') as f:
    paragraph_map['MARCO_1.txt'] = list(f.readlines())

Treccast.getParagraphInfos(paragraph_map)

