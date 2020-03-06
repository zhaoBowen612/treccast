import re
import os.path
from spacy.lang.en import English
from nltk.tokenize import sent_tokenize

# path to location of the datasets
CAR_TRECWEB_LOC = 'data/dedup.articles-paragraphs.cbor.xml'
# MARCO_TRECWEB_LOC='data/dedup.articles-paragraphs.cbor'
MARCO_TRECWEB_LOC = 'data/collection.tsv.xml'

cleanr = re.compile('<.*?>')


# divide the dataset into pieces
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


# process(MARCO_TRECWEB_LOC, "data/marco_ids")
# process(CAR_TRECWEB_LOC, "data/car_ids")


# cut paragraph into sentences
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


# reformat("data/marco_ids/")
# reformat("data/car_ids/")

nlp = English()


# cut by 20 tokens
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
                    cnt = 20
                    if len(doc) > cnt:
                        while len(doc) > cnt:
                            # print(doc[cnt - 20: cnt].text)
                            fp.write(doc[cnt - 20:cnt].text + '\n')
                            cnt += 20
                        fp.write(doc[cnt - 20:].text + '\n')
                    else:
                        fp.write(doc.text)


# cut("data/marco_test/")
# cut("data/marco_ids/")
# cut("data/car_ids/")


def check(path):
    result = []
    flag = True
    for root, dirs, files in os.walk(path):
        for file in files:
            with open(path + file, 'r', encoding='UTF-8') as fp:
                lines = fp.readlines()
                for line in lines:
                    if len(nlp(line)) > 25:
                        result.append(file)
                        flag = False
                        break
    if flag:
        print("all < 25")
    else:
        print(result)


# check('data/marco_ids/')
# check('data/car_ids/')


def test(path):
    for root, dirs, files in os.walk(path):
        # print(files)
        for file in files:
            with open(path + file, 'r', encoding='UTF-8') as fp:
                lines = fp.readlines()
            with open(path + file, 'w', encoding='UTF-8') as fp:
                for line in lines:
                    if line == '\n':
                        continue
                    else:
                        fp.write(line)


# test('data/marco_ids/')
# test('data/car_ids/')
# test('data/test_set/')
