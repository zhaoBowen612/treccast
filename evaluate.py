from bert_serving.client import BertClient
from treccast import Treccast
import os
import json
import numpy as np

ANSWER = 'data/evaluation/answer.txt'
RESULT = 'data/indri_data/indri_result/result.txt'

# load word embeddings
model = BertClient()


# model = ''


def evaluate(path):
    # thirty_queries[0:29]
    thirty_queries = get_query()
    ans, mark = get_ans()
    t = Treccast(model)
    for i in range(30):
        print('conversation', i)
        current = []
        i = 0
        for turn in thirty_queries[i]:
            print('turn', i, ':', turn)
            i += 1
            current.append(turn)
            content = {"questions": current,
                       "indriRetNbr": '10',
                       "retNbr": '10',
                       "convquery": "conv_w1",
                       "h1": '0.5',
                       "h2": '0.5',
                       }
            result_ids, para_score, result_content = t.retrieveAnswer(content)
            with open('testing.txt', 'w') as result:
                w = []
                for key, value in para_score.items():
                    w.append(key + ' : ' + str(value) + '\n')
                result.writelines(w)

    ans = open(ANSWER, 'r')
    res = open(RESULT, 'r')
    print('AP@100 is', AP(ans))
    print('nDCG@100 is', nDCG(res))
    print('ERR@100 is', ERR())
    ans.close()
    res.close()
    return


def get_query():
    thirty_queries = []
    with open('data/evaluation/train_topics_v1.0.json') as q:
        dic = json.loads(q.read())
        query = []
        for i in range(30):
            for turn in dic[i]['turn']:
                # print(turn['raw_utterance'])
                query.append(turn['raw_utterance'])
            thirty_queries.append(query)
    return thirty_queries


def get_ap_ans():
    # only relevant or not matters
    turn = dict()
    with open('data/evaluation/train_topics_mod.qrel') as ans:
        lines = ans.readlines()
        for line in lines:
            sp = line.split()
            if sp[3] != '0':
                # turn['1_2'] = "(MARCO_955948, 2)"
                turn[sp[0]] = (sp[2], sp[3])
    return turn


def AP(fp):
    ap = 0
    # turn_ans['1_2'] = "(MARCO_955948, 2)"
    turn_ans = get_ap_ans()
    cnt = 0
    # retrieved result
    result_l = []
    # actual relevant items
    rel_l = []
    for i, rel in enumerate(result_l):
        if rel in rel_l:
            cnt += 1
            ap += cnt / i + 1
    return ap / len(rel_l)


def DCG(fp):
    score = 0
    result_l = []
    for i, rel in enumerate(result_l):
        score += rel / np.log2(i + 2)
    return score


def nDCG(fp):
    dcg = DCG(fp)
    idcg = DCG(sorted(fp, reverse=True))
    return dcg / idcg


def get_ndcg_ans():
    files = []
    marks = []
    with open('data/evaluation/train_topics_mod.qrel') as ans:
        # line is 'MARCO_955948'
        files.append(ans.readline().split()[2])
        marks.append(ans.readline().split()[3])
    return files, marks


def get_ans():
    files = []
    marks = []
    with open('data/evaluation/train_topics_mod.qrel') as ans:
        lines = ans.readlines()
        for line in lines:
            sp = line.split()
            files.append(sp[2])
            marks.append(sp[3])
    return files, marks


def ERR():
    pass


evaluate('data/evaluation/')


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
