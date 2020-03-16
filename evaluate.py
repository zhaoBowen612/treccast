from bert_serving.client import BertClient
from treccast import Treccast
import os
import re
import json
import numpy as np

ANSWER = 'data/evaluation/answer.txt'
BERT_UW_RETURN = 'data/evaluation/bert_uw_return.txt'
BERT_W1_RETURN = 'data/evaluation/bert_w1_return.txt'
BERT_W2_RETURN = 'data/evaluation/bert_w2_return.txt'
CROWN_RETURN = 'data/evaluation/crown_return.txt'
RESULT = 'data/indri_data/indri_result/result.txt'

'''
# load word embeddings
model = BertClient()


def evaluate(path):
    # thirty_queries[0:29]
    thirty_queries = get_query()
    # ans, mark = get_ans()
    with open(BERT_W2_RETURN, 'w') as fp:
        for i in range(30):
            t = Treccast(model)
            print('conversation', i + 1)
            current = []
            turn_id = 0
            for turn in thirty_queries[i]:
                turn_id += 1
                print('query', i + 1, 'turn', turn)
                current.append(turn)
                content = {"questions": current,
                           "terrierRetNbr": '10',
                           "retNbr": '10',
                           # "convquery": "conv_uw",
                           # "convquery": "conv_w1",
                           "convquery": "conv_w2",
                           "h1": '0.5',
                           "h2": '0.5',
                           }
                result_ids, para_score, result_content = t.retrieveAnswer(content)
                # res := (id, mark) from high to low
                res = sorted(para_score.items(), key=lambda item: item[1], reverse=True)
                # write all return result into files
                for j in range(len(res)):
                    fp.write(str(i + 1) + '_' + str(turn_id) + ' ' + str(res[j][0]) + ' ' + str(res[j][1]) + '\n')
            # print('AP@5 is', AP())
            # print('nDCG@100 is', nDCG())
            # print('ERR@100 is', ERR())

'''


# load all queries into memory
def get_query():
    thirty_queries = []
    with open('data/evaluation/train_topics_v1.0.json') as q:
        dic = json.loads(q.read())
        for i in range(30):
            query = []
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
            print(sp[0])
            if sp[3] != '0':
                if not turn.get(sp[0]):
                    turn.update({sp[0]: [sp[2] + '.txt']})
                else:
                    turn[sp[0]].append(sp[2] + '.txt')
                # turn['1_2'] = "(MARCO_955948.txt, 2)"
                # cal AP
    return turn


def AP():
    ap = 0
    # turn_ans['1_2'] = "(MARCO_955948.txt, 2)"
    turn_ans = get_ans()
    # for k in turn_ans.keys():
    #     print(k)
    # print(turn_ans)
    # retrieved result, like
    result_l = dict()
    # ret.txt should be the output file of the 30 queries
    with open(BERT_UW_RETURN, 'r') as ret:
        lines = ret.readlines()
        for line in lines:
            sp = line.split()
            if sp[1] in turn_ans[sp[0]]:
                if not result_l.get(sp[0]):
                    result_l.update({sp[0]: [1]})
                else:
                    result_l[sp[0]].append(1)
            else:
                if not result_l.get(sp[0]):
                    result_l.update({sp[0]: [0]})
                else:
                    result_l[sp[0]].append(0)
    # actual relevant items
    rel_l = {}
    with open('data/evaluation/train.txt', 'r') as train:
        lines = train.readlines()
        for line in lines:
            sp = line.split()
            if sp[3]:
                rel_l[sp[0]].append(sp[2] + '.txt')
    sum = 0
    for k in result_l.items():
        cnt = 0
        for i, rel in enumerate(k):
            if rel in rel_l:
                cnt += 1
                ap += cnt / i + 1  # 第i个语段在相关列表中的位置 1/2 表示第一个文档在相关文档(generated)中第二位
        sum += ap / cnt
    return sum / len(result_l.items())


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
# print(AP())


# this is used to remove the answers that are not in the first 100,000 paragraphs
def rearrange(path):
    mar = []
    car = []
    has = []
    error = []

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
                if 'WAPO' not in line:
                    error.append(line.split()[2])
                continue
                # this answer.txt contain those exist locally
    with open(path + 'evaluation/answer.txt', 'w') as train:
        print(len(has))
        train.writelines(has)
    return error


missing = rearrange('data/')
