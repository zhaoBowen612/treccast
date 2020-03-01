import re
import os.path
from spacy.lang.en import English
from nltk.tokenize import sent_tokenize

# path to location of the datasets
CAR_TRECWEB_LOC = 'data/dedup.articles-paragraphs.cbor.xml'
# MARCO_TRECWEB_LOC='data/dedup.articles-paragraphs.cbor'
MARCO_TRECWEB_LOC = 'data/collection.tsv.xml'

cleanr = re.compile('<.*?>')


def process(loc, folder):
    with open(loc, 'r') as fp:
        line = fp.readline()
        id = ''
        cnt = 0
        while line:
            if "DOCNO" in line:
                id = re.sub(cleanr, '', line).replace('\n', '')
            if "<BODY>" in line:
                cnt += 1
                if cnt % 10000 == 0:
                    print(10)
                paragraph = fp.readline()
                if not os.path.isfile(folder + "/" + id + ".txt"):
                    with open(folder + "/" + id + ".txt", 'w') as out:
                        out.write(paragraph)
                        out.close()
            if cnt == 100000:
                break
            line = fp.readline()


process(MARCO_TRECWEB_LOC, "data/marco_ids")
process(CAR_TRECWEB_LOC, "data/car_ids")


def reformat(path):
    i = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            sentences = []
            print(i, ' : ', file)
            with open(path + file, 'r', encoding='UTF-8') as fp:
                for sen in sent_tokenize(fp.read().replace('\n', '')):
                    sentences.append(sen + '\n')
            with open(path + file, 'w') as fp:
                fp.writelines(sentences)
            i += 1


reformat("data/marco_ids/")
reformat("data/car_ids/")
# reformat("data/marco_test/")

nlp = English()


def cut(path):
    i = 0
    result = []
    max_len = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            sentences = []
            i += 1
            # print(i, ' : ', file)
            with open(path + file, 'r', encoding='UTF-8') as fp:
                lines = fp.readlines()
            with open(path + file, 'w', encoding='UTF-8') as fp:
                for line in lines:
                    doc = nlp(line)
                    cnt = 50
                    if len(doc) > cnt:
                        while len(doc) > cnt:
                            print(doc[cnt - 50: cnt].text)
                            fp.write(doc[cnt - 50:cnt].text + '\n')
                            cnt += 50
                        fp.write(doc[cnt - 50:].text + '\n')
                    else:
                        fp.write(doc.text)


# cut("data/marco_test/")
cut("data/marco_ids/")
cut("data/car_ids/")


def check(path):
    result = []
    flag = True
    for root, dirs, files in os.walk(path):
        for file in files:
            with open(path + file, 'r', encoding='UTF-8') as fp:
                lines = fp.readlines()
                for line in lines:
                    if len(nlp(line)) > 60:
                        result.append(file)
                        flag = False
                        break
    if flag:
        print("all < 60")
    else:
        print(result)


check('data/marco_ids/')
check('data/car_ids/')


def test(file):
    i = 0
    # result = []
    # max_len = 0
    with open(file, 'r', encoding='UTF-8') as fp:
        lines = fp.readlines()
    with open(file, 'w', encoding='UTF-8') as fp:
        for line in lines:
            doc = nlp(line)
            cnt = 50
            if len(doc) > cnt:
                while len(doc) > cnt:
                    print(doc[cnt - 50: cnt].text)
                    fp.write(doc[cnt - 50:cnt].text + '\n')
                    cnt += 50
                fp.write(doc[cnt - 50:].text + '\n')
            else:
                fp.write(doc.text)

# test('MARCO_2.txt')
