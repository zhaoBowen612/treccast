from gensim.parsing.porter import PorterStemmer
# import gensim
# import json
import os
# from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import numpy as np
# import re
from sklearn.metrics.pairwise import cosine_similarity as cs
# import networkx as nx
import operator
import math
# import itertools
# import pickle
# from argparse import ArgumentParser
import subprocess
# from subprocess import SubprocessError
import logging
from nltk.tokenize import sent_tokenize
import time

nlp = English()
stemmer = PorterStemmer()  # 词干分析器，将一词多态转化为一个词干

# location of Indri index files
CAR_INDEX_LOC = 'data/indri_data/car_index/'
MARCO_INDEX_LOC = 'data/indri_data/marco_index/'

# locations of txt files for each MARCO\CAR id
# (to create these files from the MARCO\CAR collections use preprocess_collections.py)
CAR_ID_LOC = 'data/car_ids'
MARCO_ID_LOC = 'data/marco_ids'

# location of Indri command line tool
# INDRI_LOC = 'indri-5.12\runquery'
INDRI_LOC = '/usr/local/bin'
# the co-occurence window is set to 3 for our graph
COOC_WINDOW = 3

# number of documents in MARCO TREC-CAsT corpus
# nbr_docs = 8635155
nbr_docs = 100000


class Treccast:
    def __init__(self, word_vectors):
        self.word_vectors = word_vectors
        self.call_time = time.time()
        # below is the logging part
        # self.logger = logging.getLogger("crown_logger_" + str(self.call_time))
        # self.crown_logger = logging.FileHandler("logging\Log-" + str(self.call_time) + ".log")
        # self.crown_logger.setLevel(logging.DEBUG)
        # self.logger.addHandler(self.crown_logger)
        # logging.basicConfig(filename="logging\Log-" + str(self.call_time) + ".log",level=logging.DEBUG)

    # return the embedding of each token and the tokenized queries
    # use str2word vector for query
    def getQueryEmbeddings(self, query):
        print('query:', query)
        doc = nlp(query)
        print(doc.text)
        # remaining words are meaningful ones from queries, punctuation is removed by isalpha()
        tokens = [token.text.lower() for token in doc if token.text.isalpha() and not token.is_stop]
        return self.word_vectors.encode([doc.text]), tokens  # return (bert, list)

    # get tokenized paragraph, embeddings for each token and the information in which paragraph certain token appears
    # paragraph_map is a file name to paragraph dict
    # @staticmethod
    # def getParagraphInfos(paragraph_map):
    def getParagraphInfos(self, paragraph_map):
        filename_to_embeddings = dict()  # token_embeddings[line] = vector
        # paragraph_map is dict str:str
        for key in paragraph_map:
            # key should be the file name like MARCO_1.txt
            # print('key is: ', key, 'value is: ', paragraph_map[key])
            # doc = nlp(''.join(paragraph_map[key]))
            # break one paragraph into token list
            lines = list(line for line in paragraph_map[key])
            filename_to_embeddings[key] = self.word_vectors.encode(lines)
            # tokens = list([token.text.lower() for token in doc if token.text.isalpha() and not token.is_stop])

            # paragraph_to_lines[key] = lines  # use one file name to find the related line vec
            # for l in lines:
                # get vector of each token
                # line_embeddings[l] = self.word_vectors[l]
                # if token in token_to_ids.keys():
                #     token_to_ids[token].append(key)
                # else:
                #     token_to_ids[token] = [key]
                    # token_embeddings := token to vector
                    # token_to_id := token to key(index file) list
                    # paragraph_to_tokens := key to token list
        # return token_embeddings, token_to_ids, paragraph_to_tokens
        return filename_to_embeddings

    # parse indri result file
    # returns paragraphs and its corresponding scores given by indri
    def processIndriResult(self, filename):
        with open(filename, 'r') as fp:
            # each line represent one file of car or marco
            indri_line = fp.readline()
            indri_paragraphs = dict()
            paragraph_score = dict()  # initially should store the ranking of each paragraph
            while indri_line:
                splits = indri_line.split(" ")
                splits[2] = splits[2].replace('home', 'Users')
                splits[2] = splits[2].replace('crown', 'CROWN')
                paragraph = []
                # find the related files according to each line in result indri file
                if len(splits) < 5:
                    # self.logger.warn("processed indri line has not the expected format!")
                    indri_line = fp.readline()
                    continue
                if "MARCO" in splits[2]:
                    # if os.path.exists(MARCO_ID_LOC + splits[2] + ".txt"):
                    if os.path.exists(splits[2]):
                        try:
                            print('find an marco file')
                            with open(splits[2], "r", encoding='UTF-8') as id_file:
                                # paragraph.append(line for line in id_file.readlines())
                                paragraph = id_file.readlines()
                                id_file.close()
                        except IOError:
                            continue
                    # else:
                    #     self.logger.warn("no file with this id found, id was: %s", splits[2])
                elif "CAR" in splits[2]:
                    # if os.path.exists(CAR_ID_LOC + splits[2] + ".txt"):
                    if os.path.exists(splits[2]):
                        try:
                            with open(splits[2], "r", encoding='UTF-8') as id_file:
                                print('find an car file')
                                paragraph = id_file.readlines()
                                id_file.close()
                        except IOError:
                            continue
                    else:
                        # self.logger.warn("no file with this id found, id was: %s", splits[2])
                        print("no file with this id found, id was: %s" % splits[2])
                if paragraph:
                    # splits[2] is the file name, splits[3] is the ranking 1,2,3...
                    indri_paragraphs[splits[2]] = paragraph
                    paragraph_score[splits[2]] = splits[3]
                indri_line = fp.readline()
        # indri_paragraph is filename: [lines]
        return indri_paragraphs, paragraph_score

    # creates an indri .query file using the unweighted combination of queries
    # from the current the previous and the first turn
    def createIndriQuery(self, tokens, query_tokens, turn_nbr):
        # create '.query'file first
        with open("data/indri_data/indri_queries/" + str(self.call_time) + "_turn" + str(
                turn_nbr + 1) + "_indri-query.query", "w") as query_file:
            xmlString = '''<parameters> <index>''' + MARCO_INDEX_LOC + '''</index>
                    <index>''' + CAR_INDEX_LOC + '''</index>
                    <query><number>''' + str(turn_nbr + 1) + '''</number>'''
            if turn_nbr == 0:
                # map(function, iterable argus)
                line_str = " ".join(map(str, tokens))  # 'token1 token2 token3'
                new_line = "<text>#combine(" + line_str + ")</text>"
            elif turn_nbr == 1:
                prev_data = query_tokens[turn_nbr - 1]
                line_str = " ".join(map(str, tokens)) + " " + "  ".join(map(str, prev_data))
                words = line_str.split()
                # what is 'key=words.index'
                line_str = " ".join(sorted(set(words), key=words.index))  # not sure yet
                new_line = "<text>#combine(" + line_str + ")</text>"
            else:
                prev_data = query_tokens[turn_nbr - 1]
                first_data = query_tokens[0]
                line_str = " ".join(map(str, tokens)) + " " + " ".join(map(str, prev_data)) + " " + " ".join(
                    map(str, first_data))
                words = line_str.split()
                line_str = " ".join(sorted(set(words), key=words.index))
                new_line = "<text>#combine(" + line_str + ")</text>"
            xmlString += new_line
            xmlString += '''</query></parameters>'''
            print('query is: ' + xmlString)
            query_file.write(xmlString)

    # main answering function: receives all relevant parameters to answer the request
    # returns the highest scoring paragraphs
    def retrieveAnswer(self, parameters):  # parameter here is a json

        # read in the parameters, these parameters are ot from input json
        conv_queries = parameters["questions"]
        turn_nbr = len(conv_queries) - 1
        INDRI_RET_NUM = int(parameters["indriRetNbr"])
        EDGE_THRESHOLD = float(parameters["edgeThreshold"])
        NODE_MATCH_THRESHOLD = float(parameters["nodeThreshold"])
        res_nbr = int(parameters["retNbr"])
        convquery_type = parameters["convquery"]
        h1 = float(parameters["h1"])
        h2 = float(parameters["h2"])

        # store the vector of each question\turn
        turn_query_embeddings = dict()
        query_tokens = dict()

        # get tokenized queries and the embeddings of each token
        for i in range(len(conv_queries)):  # traverse each query and get vectors and token lists
            # for each query, return dic(tokens in one query, embedding), list
            # why this line is wrong
            # turn_query_embeddings[i], query_tokens[i] = self.getQueryEmbeddings(conv_queries[i])
            tmp_tokens = self.getQueryEmbeddings(conv_queries[i])
            # turn_query_embeddings[i] = tmp_embeddings
            query_tokens[i] = tmp_tokens

        # turn_nbr := Number of questions
        # in such a situation, we only consider the last one as the current query?
        current_query_embeddings = turn_query_embeddings[turn_nbr]  # get the tokens embeddings of current query
        # tokens = query_tokens[turn_nbr]  # get the tokens of the current turn
        query_turn_weights = dict()
        # dict(tokens of chosen queries, embeddings)
        conv_query_embeddings = dict(current_query_embeddings)  # current turn is necessary for the three options

        # create the conversational query (3 options are available)
        if convquery_type == "conv_uw":  # option 1: current + first
            if turn_nbr != 0:  # as long as the current is not the first one
                #  add embeddings of the tokens in the first query
                conv_query_embeddings.update(turn_query_embeddings[0])
        elif convquery_type == "conv_w1":  # option 2: current + previous + first
            if turn_nbr != 0:
                conv_query_embeddings.update(turn_query_embeddings[0])  # first
            for token in conv_query_embeddings.keys():
                query_turn_weights[token] = 1.0
            if turn_nbr > 1:
                for token in turn_query_embeddings[turn_nbr - 1].keys():
                    if not token in query_turn_weights.keys():  # avoid duplicates
                        query_turn_weights[token] = turn_nbr / (turn_nbr + 1)  # weights can be found in the essays
                conv_query_embeddings.update(turn_query_embeddings[turn_nbr - 1])  # previous
        else:
            # convquery_type == "conv_w2":  # all queries are considered
            query_turn_weights = dict()  # token to related query turn weight
            for j in range(turn_nbr + 1):
                t_embeddings = turn_query_embeddings[j]
                for token in t_embeddings.keys():
                    conv_query_embeddings[token] = t_embeddings[token]
                    if j == 0 or j == turn_nbr:
                        query_turn_weights[token] = 1.0
                    else:
                        if token in query_turn_weights.keys():
                            if query_turn_weights[token] == 1.0:
                                continue
                        query_turn_weights[token] = (j + 1) / (turn_nbr + 1)

        # create Indri query
        # tokens are all tokens in the current query,
        # query_tokens[0] := all tokens of the first query,
        # turn_nbr := the num of the current query
        self.createIndriQuery(tokens, query_tokens, turn_nbr)
        # self.logger.info("indri query created successfully")

        # do indri search

        print('sending query')
        subprocess.run(['scp', 'data/indri_data/indri_queries/indri-query.query',
                        'zhaobowen@192.168.176.142:~/PycharmProjects/crown/data/indri_data/indri_queries/'])
        while not os.path.exists('data/indri_data/indri_results/result.txt'):
            time.sleep(1)

            '''
            subprocess.run([INDRI_LOC + "/IndriRunQuery",
                            "data/indri_data/indri_queries/" + str(self.call_time) + "_turn" + str(
                                turn_nbr + 1) + "_indri-query.query", "-count= " + str(INDRI_RET_NUM),
                            "-trecFormat=true"], stdout=outfile)
            '''

        # prepare indri paragraphs: get paragraph sentences and original indri scores from indri result file
        # indri_patagraph_score is file name to ranking. get reciprocal as the Indri mark
        indri_paragraphs, indri_paragraph_score = self.processIndriResult(
            # 'data/indri_data/indri_results/result' + "_" + str(self.call_time) + "_turn" + str(turn_nbr + 1) + '.txt')
            'data/indri_data/indri_results/result.txt')

        # get tokenized paragraphs, its token embeddings and info which token belongs to which paragraph

        # token_embeddings := token to vector
        # token_to_id := token to key(index file) list
        # paragraph_to_tokens := key to token list
        # indri_paragraphs is a name to paragraph dict
        line_embeddings = self.getParagraphInfos(indri_paragraphs)

        # calculate our indri_score which is 1 / indri rank
        for id in indri_paragraph_score.keys():
            indri_paragraph_score[id] = 1 / int(indri_paragraph_score[id])

'''
        query_to_graph_token = dict()
        # calculate node weights -> note: there are tokens which do not have an embedding: node weight=0
        # start calculating another two scores of each candidate paragraph
        for p_token in token_to_ids.keys():  # token_to_ids is token to file name
            max_sim = 0.0
            max_q_token = ''
            if p_token in token_embeddings.keys():  # if this token has a related vector
                for q_token in conv_query_embeddings.keys():
                    # calculate the cos between query tokens and passages tokens
                    # representation?????
                    [[sim]] = cs([token_embeddings[p_token]], [conv_query_embeddings[q_token]])
                    #  get the max_sim and the related query token
                    # used in essay for node score
                    if sim > max_sim:
                        max_sim = sim
                        max_q_token = q_token
'''
# divide

'''
            # calculate edge weights
            for k in range(len(p_tokens)):
                if not p_tokens[k] in self.G.nodes():  # still is this line necessary?
                    continue
                # check if token is close enough to a conversational query token (> NODE_MATCH_THRESHOLD)
                if not p_tokens[k] in query_to_graph_token.keys():
                    # query_to_graph_token is from p_token to find max q_tokens
                    continue
                if k >= (len(p_tokens) - COOC_WINDOW):  # context windows???
                    upper_3 = len(p_tokens)
                else:
                    upper_3 = k + COOC_WINDOW + 1
                    # go over all tokens which are in proximity 3 to current token
                for j in range(k + 1, upper_3):
                    if not p_tokens[j] in self.G.nodes():
                        continue
                    if p_tokens[j] == p_tokens[k]:
                        continue
                    if not p_tokens[j] in query_to_graph_token.keys():
                        continue
                    t1 = p_tokens[k]
                    t2 = p_tokens[j]
                    # check if there is an edge between the two
                    if t1 in self.G.adj[t2]:
                        # check if the two token are not similar to the same query token
                        if not np.intersect1d(query_to_graph_token[t1], query_to_graph_token[t2]):
                            # note that the current edge weight in the graph is the pmi value,
                            # here nmpi is calculated out of it
                            edge_weight = self.G.get_edge_data(t1, t2)['weight']
                            if t1 < t2:
                                prox3_prob = self.prox_dict[str(t1) + "_" + str(t2)] / nbr_docs
                            else:
                                prox3_prob = self.prox_dict[str(t2) + "_" + str(t1)] / nbr_docs
                                #  calculate npmi
                            edge_weight /= (- math.log(prox3_prob, 2))
                            # consider edge weight if it is above the edge threshold
                            if edge_weight > EDGE_THRESHOLD:
                                edge_score += edge_weight
                                edge_count += 1
                                if t1 < t2:
                                    pair = "(" + str(t1) + "," + str(t2) + ")"
                                else:
                                    pair = "(" + str(t2) + "," + str(t1) + ")"
                                edge_map[p_key].append(pair)
                                if not pair in edge_weight_map.keys():
                                    edge_weight_map[pair] = edge_weight
            # calculate the final edge score for the current paragraph
            if edge_count != 0:
                edge_score = edge_score / edge_count
            edge_score_dict[p_key] = edge_score
'''
# self.logger.info("node and edge scores are calculated")

'''
# combine scores
for p_key in indri_paragraphs.keys():
    if not p_key in indri_paragraph_score.keys():
        indri_paragraph_score[p_key] = 0.0

for p_key in indri_paragraphs.keys():
    p_score = h1 * float(indri_paragraph_score[p_key]) + h2 * node_score_dict[p_key] + h3 * edge_score_dict[
        p_key]
    scored_paragraphs_dict[p_key] = p_score

# sort paragraphs according to three scores
sorted_p = sorted(scored_paragraphs_dict.items(), key=operator.itemgetter(1), reverse=True)
scored_paragraphs = [x[0] for x in sorted_p]

# sort node and edge token candidates
for p_key in indri_paragraphs.keys():
    node_map[p_key] = list(set(node_map[p_key]))
    node_map[p_key] = sorted(node_map[p_key], key=lambda x: self.G.nodes[x]['weight'], reverse=True)
    if len(node_map[p_key]) > 5:
        del node_map[p_key][5:]
    edge_map[p_key] = list(set(edge_map[p_key]))
    edge_map[p_key] = sorted(edge_map[p_key], key=lambda x: edge_weight_map[x], reverse=True)
    if len(edge_map[p_key]) > 5:
        del edge_map[p_key][5:]

# get final list of paragraphs that will be returned to the user
result_paragraphs = []
result_ids = []
result_node_map = dict()
result_edge_map = dict()
for p in range(len(scored_paragraphs)):
    if p < res_nbr:  # res_nbr is number of passages return finally
        result_paragraphs.append(indri_paragraphs[scored_paragraphs[p]])
        result_ids.append(scored_paragraphs[p])
        #     self.logger.info("Top : %i", (p+1))
        #     self.logger.info("Paragraph ID: %s, score: %s", scored_paragraphs[p], sorted_p[p][1])
        #     self.logger.info("Paragraph: %s", indri_paragraphs[scored_paragraphs[p]])
        # else:
        break

for res_id in result_ids:
    result_node_map[res_id] = node_map[res_id]
    result_edge_map[res_id] = edge_map[res_id]

return result_paragraphs, result_ids, result_node_map, result_edge_map
'''
