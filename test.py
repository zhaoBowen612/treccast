from treccast import *
from sklearn.metrics.pairwise import cosine_similarity as cs
import re
import nltk
from spacy.lang.en import English

nlp = English()

paragraph_map = {}

has = dict()
has1 = {}
CAR_TRECWEB_LOC = 'data/dedup.articles-paragraphs.cbor.xml'
MARCO_TRECWEB_LOC = 'data/collection.tsv.xml'

"""
    iterable: 待排序的可迭代集合
    key: iterable中每一项中的第几个key
    reverse: true 降序，false 升序
"""
# res = sorted(dict_.items(), key=lambda item: item[1], reverse=True)
# for key in dict_.keys():
#     print(key, dict_[key])
# print(res)

dic = {1: 2, 2: 3, 3: 4}
print(dic[4])
