from treccast import *
from sklearn.metrics.pairwise import cosine_similarity as cs
import re
import nltk
from spacy.lang.en import English

nlp = English()

paragraph_map = {}


# with open('MARCO_3.txt', 'r+') as f:
#     print(f.readlines())
#     f.write('123')

#     paragraph_map['MARCO_1.txt'] = list(f.readlines())
#     lines = f.readlines()
#     for line in lines:
#         print(list(nlp(line)))
#         print(len(nlp(line)))
# Treccast.getParagraphInfos(paragraph_map)


def count(path):
    i = 0
    result = []
    max_len = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            sentences = []
            i += 1
            print(i, ' : ', file)
            with open(path + file, 'r', encoding='UTF-8') as fp:
                lines = fp.readlines()
                for line in lines:
                    cnt_len = len(nlp(line))
                    # print(i, cnt_len)
                    if max_len < cnt_len:
                        max_len = cnt_len
                        if cnt_len > 100:
                            result.append(file)

    for file in result:
        print('processing', file)
        with open(path + file, 'r', encoding='UTF-8') as fp:
            lines = fp.readlines()
        with open(path + file, 'w', encoding='UTF-8') as fp:
            add = []
            for line in lines:
                add += list(filter(None, re.split('\.|\!|\?|\n|\:|\;|\,', line)))
            fp.writelines(add)
    print(result)
    print('max length of a line is: ', max_len)


# count("data/car_ids/")
# count("data/marco_ids/")

doc = nlp('how are you I am fine thank you and you')
# print(nlp)
# print(doc[0: 3])
# print(doc[3: 6])
# print(doc[6: 9])

# with open('/Users/zhaobowen/PycharmProjects/CROWN/data/marco_ids/MARCO_71485.txt', 'r') as fp:
#     lines = fp.readlines()
#     for line in lines:
#         print(line)
#     print(len(nlp(fp.readline())))
#     print(len(nlp(fp.readline())))
# print(cs([[1, 2, 3]], [[1, 2, 4]]))
# [[sim]] = cs([[1, 2, 3]], [[1, 2, 4]])
# print(sim)
# print([[sim]])
# aList = [123,  'zara', 'abc']
#
# aList.remove('xyz')
# print(aList)
li = [123, 234, 345, 456, 567, 678, 789]
for i in range(len(li)):
    print(i, li[i])
