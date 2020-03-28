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
CROWN_UW_RETURN = 'data/evaluation/crown_uw_return.txt'
CROWN_W1_RETURN = 'data/evaluation/crown_w1_return.txt'
CROWN_W2_RETURN = 'data/evaluation/crown_w2_return.txt'
RESULT = 'data/indri_data/indri_result/result.txt'

# load word embeddings
# model = BertClient()

'''
def evaluate(path):
    # thirty_queries[0:29]
    thirty_queries = get_query()
    # ans, mark = get_ans()
    with open(BERT_UW_RETURN, 'w') as fp:
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
                           "retNbr": '20',
                           "convquery": "conv_uw",
                           # "convquery": "conv_w1",
                           # "convquery": "conv_w2",
                           "h1": '0.5',
                           "h2": '0.5',
                           }
                result_ids, para_score, result_content = t.retrieveAnswer(content, attention=False)
                # res := (id, mark) from high to low
                res = sorted(para_score.items(), key=lambda item: item[1], reverse=True)
                # write all return result into files
                for j in range(len(res)):
                    fp.write(str(i + 1) + '_' + str(turn_id) + ' ' + str(res[j][0]) + ' ' + str(res[j][1]) + '\n')
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


def get_ap_ans():
    # only relevant or not matters
    turn = dict()
    with open(ANSWER) as ans:
        lines = ans.readlines()
        for line in lines:
            sp = line.split()
            if sp[3] != '0':
                if not turn.get(sp[0]):
                    turn.update({sp[0]: [sp[2] + '.txt']})
                else:
                    turn[sp[0]].append(sp[2] + '.txt')
                # turn['1_2'] = "(MARCO_955948.txt, 2)"
                # cal AP
    return turn


def AP(comp):
    ap = 0
    # turn_ans['1_2'] = "(MARCO_955948.txt, 2)"
    turn_ans = get_ap_ans()
    # for k in turn_ans.keys():
    #     print(k)
    # print(turn_ans)
    # retrieved result, like
    result_l = dict()
    # ret.txt should be the output file of the 30 queries
    with open(comp, 'r') as ret:
        lines = ret.readlines()
        for line in lines:
            sp = line.split()
            try:
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
            except KeyError:
                # print(sp[0])
                pass
    # for key in result_l.keys():
    #     print(key, result_l[key])
    # actual relevant items
    rel_l = {}
    with open('data/evaluation/answer.txt', 'r') as train:
        lines = train.readlines()
        for line in lines:
            sp = line.split()
            if sp[3]:
                if not rel_l.get(sp[0]):
                    rel_l.update({sp[0]: []})
                else:
                    rel_l[sp[0]].append(sp[2] + '.txt')
    res = 0
    for k in result_l.keys():
        # print(result_l[k])
        cnt = 0
        ap = 0
        # rel is 0 or 1
        for i, rel in enumerate(result_l[k]):
            if not rel:
                cnt += 1
                ap += cnt / (i + 1)  # 第i个语段在相关列表中的位置 1/2 表示第一个文档在相关文档(generated)中第二位
        if cnt:
            res += ap / cnt
    # calculate average AP of each turn
    return res / len(result_l.keys())


def get_dcg_ans():
    # consider rank of relevance
    turn = dict()
    with open(ANSWER) as ans:
        lines = ans.readlines()
        for line in lines:
            sp = line.split()
            if sp[3] != '0':
                if not turn.get(sp[0]):
                    turn.update({sp[0]: {sp[2]: sp[3]}})
                else:
                    turn[sp[0]].update({sp[2]: sp[3]})
                # turn['1_2'] = [{'file1': 2, :, :}]
    return turn


# get_output should return rel_l['1_1] = [2,0,1,2,0,0]
def get_output(path):
    rel_l = {}
    # result_l is from training set
    result_l = get_dcg_ans()
    with open(path, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            sp = line.split()
            # first whether the key '1_1' existed
            if not rel_l.get(sp[0]):
                rel_l.update({sp[0]: []})
                # whether the search result is in the training set
                try:
                    if result_l[sp[0]].get(sp[1].replace('.txt', '')):
                        rel_l[sp[0]].append(int(result_l[sp[0]].get(sp[1].replace('.txt', ''))))
                    else:
                        # not in the training set mean 0 relevance
                        rel_l[sp[0]].append(0)
                except KeyError:
                    pass
            # key '1_1' is existed
            else:
                try:
                    # whether the search result is in the training set
                    if result_l[sp[0]].get(sp[1].replace('.txt', '')):
                        rel_l[sp[0]].append(int(result_l[sp[0]].get(sp[1].replace('.txt', ''))))
                    else:
                        # not in the training set mean 0 relevance
                        rel_l[sp[0]].append(0)
                except KeyError:
                    pass
        return rel_l


def cal_dcg(path):
    res = 0
    cnt = 0
    # rel_l is from the output of the algorithm
    rel_l = get_output(path)
    for k in rel_l.keys():
        # print(k, rel_l[k])
        res += nDCG(rel_l[k])
        cnt += 1
    return res / cnt


def DCG(seq):
    score = 0
    for i, rel in enumerate(seq):
        score += pow(2, rel) / np.log2(i + 2)  # (i + 1) + 1
        # score += rel / np.log2(i + 2)  # (i + 1) + 1
    return score


def nDCG(seq):
    # seq should be sequence of the ranks like 2 1 2 0 0 1
    dcg = DCG(seq)
    # IDCG should be sorted version of ranks
    idcg = DCG(sorted(seq, reverse=True))
    if idcg:
        return dcg / idcg
    else:
        return 0


def get_ndcg_ans():
    files = []
    marks = []
    with open(ANSWER) as ans:
        # line is 'MARCO_955948'
        files.append(ans.readline().split()[2])
        marks.append(ans.readline().split()[3])
    return files, marks


# def ERR():
#     pass


# evaluate('data/evaluation/')
print(cal_dcg(BERT_UW_RETURN))
print(cal_dcg(BERT_W1_RETURN))
print(cal_dcg(BERT_W2_RETURN))
print(cal_dcg(CROWN_UW_RETURN))
print(cal_dcg(CROWN_W1_RETURN))
print(cal_dcg(CROWN_W2_RETURN))

print(AP(BERT_UW_RETURN))
print(AP(BERT_W1_RETURN))
print(AP(BERT_W2_RETURN))
print(AP(CROWN_UW_RETURN))
print(AP(CROWN_W1_RETURN))
print(AP(CROWN_W2_RETURN))


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

# missing = rearrange('data/')
