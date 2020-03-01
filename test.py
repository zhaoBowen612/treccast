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
lines = ['The bombing of Hiroshima and Nagasaki.\n', 'On August 6, 1945 the US dropped an atomic bomb (Little Boy) on Hiroshima in Japan.\n', 'Three days later a second atomic bomb (Fat Man) was dropped on the city of Nagasaki.These were the only times nuclear weapons have been used in war.Reasons for the bombing.Many reasons are given as to why the US administration decided to drop\n', 'the atomic bomb on Hiroshima and Nagasaki.\n', '\n', "Reasons include the following: 1  The United States wanted to force Japan's surrender as quickly as possible to minimize American casualties.easons for the bombing.\n", 'Many reasons are given as to why the US administration decided to drop the atomic bomb on Hiroshima and Nagasaki.\n', "Reasons include the following: 1  The United States wanted to force Japan's surrender as quickly as possible to minimize American casualties.\n"]
l = []
print(lines)
for line in lines:
    line = line.strip('\n')
    line = line.strip(' ')
    l.append(line)
print(list(filter(None, l)))