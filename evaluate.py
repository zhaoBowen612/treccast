from bert_serving.client import BertClient
from treccast import Treccast
import os
import json
import numpy as np

ANSWER = 'data/evaluation/answer.txt'
RESULT = 'data/indri_data/indri_result/result.txt'

# load word embeddings
model = BertClient()


def evaluate(path):
    # thirty_queries[0:29]
    thirty_queries = get_query()
    # ans, mark = get_ans()
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
                       "convquery": "conv_uw",
                       # "convquery": "conv_w1",
                       # "convquery": "conv_w2",
                       "h1": '0.5',
                       "h2": '0.5',
                       }
            result_ids, para_score, result_content = t.retrieveAnswer(content)
            # res := (id, mark) from high to low
            res = sorted(para_score.items(), key=lambda item: item[1], reverse=True)
            # write all return result into files
            with open(ANSWER, 'w') as fp:
                for j in range(len(res)):
                    fp.write(res[j][0] + ' ' + res[j][1] + '\n')
            print('AP@5 is', AP())
            # print('nDCG@100 is', nDCG())
            # print('ERR@100 is', ERR())

    ans = open(ANSWER, 'r')
    res = open(RESULT, 'r')
    ans.close()
    res.close()
    return


# load all queries into memory
def get_query():
    thirty_queries = []
    with open('data/evaluation/train_topics_v1.0.json') as q:
        dic = json.loads(q.read())
        query = []
        for i in range(30):
            for turn in dic[i]['turn']:
                query.append(turn['raw_utterance'])
            thirty_queries.append(query)
    return thirty_queries


def get_ans():
    # only relevant or not matters
    turn = dict()
    with open(ANSWER) as ans:
        lines = ans.readlines()
        for line in lines:
            sp = line.split()
            if sp[3] != '0':
                # turn['1_2'] = "(MARCO_955948, 2)"
                turn[sp[0]] = (sp[2], sp[3])
    return turn


def AP():
    ap = 0
    # turn_ans['1_2'] = "(MARCO_955948, 2)"
    turn_ans = get_ans()
    cnt = 0
    # retrieved result, like
    result_l = []
    # ret.txt should be the output file of the 30 queries
    with open('data/evaluation/answer.txt', 'r') as ret:
        lines = ret.readlines()
        for line in lines:
            if turn_ans[][line.split()[]]:
                result_l.append(1)
            else:
                result_l.append(0)
    # actual relevant items
    rel_l = []
    with open('data/evaluation/train.txt', 'r') as train:
        lines = train.readlines()
        for line in lines:
            sp = line.split()
            if sp[3]:
                rel_l.append(sp[2] + '.txt')
    for i, rel in enumerate(result_l):
        if rel in rel_l:
            cnt += 1
            ap += cnt / i + 1  # 第i个语段在相关列表中的位置 1/2 表示第一个文档在相关文档中第二位
    return ap / len(rel_l)


def DCG(fp):
    score = 0
    result_l = []
    for i, rel in enumerate(result_l):
        score += rel / np.log2(i + 2)
    return score


def nDCG(fp):
    # fp should be sequence of the ranks like 2 1 2 0 0 1
    dcg = DCG(fp)
    # IDCG should be sorted version of ranks
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


# def ERR():
#     pass


# evaluate('data/evaluation/')


# this is used to remove the answers that are not in the first 100,000 paragraphs
def rearrange(path):
    mar = []
    car = []
    has = []

    # store the filename of the first 100,000
    for root, dirs, files in os.walk(path + 'marco_ids'):
        for file in files:
            mar.append(file.replace('.txt', ''))
    for root, dirs, files in os.walk(path + 'car_ids'):
        for file in files:
            car.append(file.replace('.txt', ''))

    # remove the answer that are not in first 100,000
    with open(path + 'evaluation/train_topics_mod.qrel', 'r') as ori:
        lines = ori.readlines()
        for line in lines:
            # print(line.split()[2])
            name = line.split()[2]
            if 'CAR' in name and name in car:
                has.append(line)
            elif 'MAR' in name and name in mar:
                has.append(line)
            else:
                # print('Error', line)
                continue
    with open(path + 'evaluation/train.txt', 'w') as train:
        print(len(has))
        train.writelines(has)


rearrange('data/')
