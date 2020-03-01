from treccast import *
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
'''
['CAR_008793e947ff278a4016db79b3280045b881165f.txt', 'CAR_00899d0eeb9a627550940ef58f22432d13c7c869.txt'] 
'''
special = ['CAR_00d98b5c3c5a1d15c744598d30b0086df076e4cd.txt',
           'CAR_005b665f1603883f374c0b62337f960f7f0ca6bf.txt',
           'CAR_000cfbd578301188c4030f31c1a7498aadf013b4.txt',
           'CAR_0052df8c4ffa18d8093901f7d05672ff50b8385e.txt',
           'CAR_001be17cf3c11f2cff765dff17296c6739a3b75e.txt',
           'CAR_005efe0d746119f4cd4aad942b72acea9872352c.txt']

# count("data/marco_ids/")

doc = nlp('how are you I am fine thank you and you')
print(nlp)
print(doc[0: 3])
print(doc[3: 6])
print(doc[6: 9])

