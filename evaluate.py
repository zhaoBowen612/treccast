from bert_serving.client import BertClient
from treccast import Treccast
import os
import json

ANSWER = 'data/evaluation/answer.txt'
RESULT = 'data/indri_data/indri_result/result.txt'

# load word embeddings
# model = BertClient()
model = ''


def evaluate(query_num):
    # thirty_queries[0:29]
    thirty_queries = get_query()
    ans, mark = get_ans()
    content = {"questions": [],
               "indriRetNbr": 10,
               "retNbr": 5,
               "convquery": "conv_w1",
               "h1": 0.5,
               "h2": 0.5,
               }
    pass
    t = Treccast(model)
    t.retrieveAnswer(query)
    ans = open(ANSWER, 'r')
    res = open(RESULT, 'r')
    print('AP@100 is', AP(ans))
    print('nDCG@100 is', nDCG(res))
    print('ERR@100 is', ERR())
    ans.close()
    res.close()
    pass


def get_query():
    thirty_queries = []
    with open('data/evaluation/train_topics_v1.0.json') as q:
        dic = json.loads(q.read())
        query = []
        for turn in dic[0]['turn']:
            query.append(turn['raw_utterance'])
        thirty_queries.append(query)
    return thirty_queries


get_query()


def get_ans():
    files = []
    marks = []
    with open('data/evaluation/train_topics_mod.qrel') as ans:
        # line is 'MARCO_955948'
        files.append(ans.readline().split()[2])
        marks.append(ans.readline().split()[3])
    return files, marks


def ERR():
    pass


def AP(fp):
    pass


def nDCG(fp):
    pass


# evaluate('data/evaluation/')


# this is used to remove the answers that are not in the first 100,000 paragraphs
def rearrange(path):
    mar = []
    car = []
    has = []

    # store the filename of the first 100,000
    for root, dirs, files in os.walk(path + 'marco_ids'):
        for file in files:
            mar.append(file)

    for root, dirs, files in os.walk(path + 'car_ids'):
        for file in files:
            car.append(file)

    # remove the answer that are not in first 100,000
    with open(path + 'evaluation/train_topics_mod.qrel', 'r') as ori:
        lines = ori.readlines()
        for line in lines:
            name = line.split()[2]
            if 'CAR' in name:
                if name in car:
                    has.append(line)
            elif 'MAR' in name:
                if name in mar:
                    has.append(line)
            else:
                print('Error', line)
    with open(path + 'evaluation/answer.txt', 'w') as ans:
        ans.writelines(lines)

# rearrange('data/')
