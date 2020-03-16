from gensim.parsing.porter import PorterStemmer
import os
from spacy.lang.en import English
from sklearn.metrics.pairwise import cosine_similarity
import time
import subprocess

nlp = English()
stemmer = PorterStemmer()  # 词干分析器，将一词多态转化为一个词干

# location of terrier index files
# CAR_INDEX_LOC = 'data/terrier_data/car_index/'
# MARCO_INDEX_LOC = 'data/terrier_data/marco_index/'

# locations of txt files for each MARCO\CAR id
# (to create these files from the MARCO\CAR collections use preprocess_collections.py)
CAR_ID_LOC = 'data/car_ids/'
MARCO_ID_LOC = 'data/marco_ids/'

# location of terrier command line tool
TERRIER_LOC = '/usr/local/terrier-project-5.2/'

# number of documents in MARCO TREC-CAsT corpus
# nbr_docs = 8635155
nbr_docs = 100000


class Treccast:
    def __init__(self, word_vectors):
        self.word_vectors = word_vectors
        self.call_time = time.time()

    # return the embedding of each token and the tokenized queries
    # use str2word vector for query
    def getQueryEmbeddings(self, query):
        # TODO: consider to remove the punctuation before BERTing
        # print('query:', query)
        doc = nlp(query)
        # remaining words are meaningful ones from queries, punctuation is removed by isalpha()
        tokens = [token.text.lower() for token in doc if token.text.isalpha() and not token.is_stop]
        return self.word_vectors.encode([doc.text]), tokens  # return (bert, list)

    # get tokenized paragraph, embeddings for each token and the information in which paragraph certain token appears
    # paragraph_map is a file name to paragraph dict
    def getParagraphInfos(self, paragraph_map):
        filename_to_embeddings = dict()
        # paragraph_map is dict str:str
        for key in paragraph_map:
            # key is the file name like MARCO_1.txt
            # break one paragraph into sentences list
            lines = list(line for line in paragraph_map[key])
            filename_to_embeddings[key] = self.word_vectors.encode(lines)
        return filename_to_embeddings

    # parse terrier result file
    # returns paragraphs and its corresponding scores given by terrier
    def processTerrierResult(self, filename):
        with open(filename, 'r') as fp:
            # each line represent one file of car or marco
            cnt = 0
            terrier_line = fp.readline()
            cnt += 1
            terrier_paragraphs = dict()
            paragraph_score = dict()  # initially should store the ranking of each paragraph
            while terrier_line:
                splits = terrier_line.split(" ")
                paragraph = []
                # find the related files according to each line in result terrier file
                if len(splits) < 5:
                    print("processed terrier line has not the expected format!")
                    terrier_line = fp.readline()
                    continue
                if "MARCO" in splits[2]:
                    if os.path.exists(MARCO_ID_LOC + splits[2]):
                        try:
                            with open(MARCO_ID_LOC + splits[2], "r", encoding='UTF-8') as id_file:
                                paragraph = id_file.readlines()
                                id_file.close()
                        except IOError:
                            continue
                    else:
                        print("no file with this id found, id was: %s" % splits[2])
                elif "CAR" in splits[2]:
                    if os.path.exists(CAR_ID_LOC + splits[2]):
                        try:
                            with open(CAR_ID_LOC + splits[2], "r", encoding='UTF-8') as id_file:
                                paragraph = id_file.readlines()
                                id_file.close()
                        except IOError:
                            continue
                    else:
                        print("no file with this id found, id was: %s" % splits[2])
                if paragraph:
                    # splits[2] is the file name, splits[3] is the ranking 1,2,3...
                    terrier_paragraphs[splits[2]] = paragraph
                    paragraph_score[splits[2]] = int(splits[3]) + 1
                if cnt == 20:
                    break
                terrier_line = fp.readline()
                cnt += 1
        # terrier_paragraph is filename: [lines]
        return terrier_paragraphs, paragraph_score

    # creates an terrier .query file using the unweighted combination of queries
    # from the current the previous and the first turn
    def createTerrierQuery(self, tokens, query_tokens, turn_nbr):
        # create '.query'file first
        with open("data/terrier_data/terrier_queries/" + str(self.call_time) + "_turn" + str(
                turn_nbr + 1) + "_terrier-query.xml", "w") as query_file:
            xmlString = '<TOP><NUM>001</NUM><TITLE>'

            if turn_nbr == 0:
                # map(function, iterable argus)
                line_str = " ".join(map(str, tokens))  # 'token1 token2 token3'
                new_line = line_str
            elif turn_nbr == 1:
                prev_data = query_tokens[turn_nbr - 1]
                line_str = " ".join(map(str, tokens)) + " " + "  ".join(map(str, prev_data))
                words = line_str.split()
                # what is 'key=words.index'
                line_str = " ".join(sorted(set(words), key=words.index))  # not sure yet
                new_line = line_str
            else:
                prev_data = query_tokens[turn_nbr - 1]
                first_data = query_tokens[0]
                line_str = " ".join(map(str, tokens)) + " " + " ".join(map(str, prev_data)) + " " + " ".join(
                    map(str, first_data))
                words = line_str.split()
                line_str = " ".join(sorted(set(words), key=words.index))
                new_line = line_str
            xmlString += new_line
            xmlString += '</TITLE></TOP>'
            query_file.write(xmlString)

    # main answering function: receives all relevant parameters to answer the request
    # returns the highest scoring paragraphs
    def retrieveAnswer(self, parameters):  # parameter here is a json
        # read in the parameters, these parameters are ot from input json
        conv_queries = parameters["questions"]
        turn_nbr = len(conv_queries) - 1
        terrier_RET_NUM = int(parameters["terrierRetNbr"])
        res_nbr = int(parameters["retNbr"])
        convquery_type = parameters["convquery"]
        h1 = float(parameters["h1"])
        h2 = float(parameters["h2"])

        # store the vector of each question\turn
        turn_query_embeddings = dict()
        query_tokens = dict()

        # traverse each query and get vectors and token lists
        for i in range(len(conv_queries)):
            # for each query, return dic(line, embedding), list
            turn_query_embeddings[i], query_tokens[i] = self.getQueryEmbeddings(conv_queries[i])

        # turn_nbr := Number of questions - 1
        # in such a situation, we only consider the last one as the current query
        first_query_embeddings = turn_query_embeddings[0]  # get the line_embeddings of the first query
        tokens = query_tokens[0]  # get the tokens of the the first turn
        conv_query_embeddings = [first_query_embeddings]  # current turn is necessary for the three options
        # create the conversational query (3 options are available)
        if convquery_type == "conv_uw":  # option 1: current + first
            if turn_nbr != 0:  # as long as the current is not the first one
                #  add embeddings of the tokens in the first query
                conv_query_embeddings.append(turn_query_embeddings[turn_nbr])
        elif convquery_type == "conv_w1":  # option 2: current + previous + first
            if turn_nbr != 0:
                conv_query_embeddings.append(turn_query_embeddings[turn_nbr - 1])
            if turn_nbr > 1:
                # for token in turn_query_embeddings[turn_nbr - 1].keys():
                # query_turn_weights[token] = turn_nbr / (turn_nbr + 1)  # weights can be found in the essays
                conv_query_embeddings.append(turn_query_embeddings[turn_nbr])  # previous
        else:
            # convquery_type == "conv_w2":  # all queries are considered
            # query_turn_weights = dict()  # token to related query turn weight
            # conv_query_embeddings.clear()
            # print('conv_query_embeddings', conv_query_embeddings)
            for i in range(turn_nbr + 1):
                conv_query_embeddings.append(turn_query_embeddings[i])
            #     conv_query_embeddings[token] = t_embeddings[token]
            #     if j == 0 or j == turn_nbr:
            # query_turn_weights[token] = 1.0
            # else:
            #     if token in query_turn_weights.keys():
            #         if query_turn_weights[token] == 1.0:
            #             continue
            #     query_turn_weights[token] = (j + 1) / (turn_nbr + 1)

        # create terrier query
        # tokens are all tokens in the current query,
        # query_tokens[0] := all tokens of the first query,
        # turn_nbr := the index of the current query
        self.createTerrierQuery(tokens, query_tokens, turn_nbr)

        # do terrier search
        subprocess.run([TERRIER_LOC + "/bin/terrier", "batchretrieve", "-t",
                        "data/terrier_data/terrier_queries/" + str(self.call_time) + "_turn" + str(
                            turn_nbr + 1) + "_terrier-query.xml"], stdout=subprocess.PIPE)
        os.rename('data/terrier_data/terrier_results/results.res',
                  'data/terrier_data/terrier_results/result' + "_" + str(self.call_time) + "_turn" + str(
                      turn_nbr + 1) + '.txt')

        # prepare terrier paragraphs: get paragraph sentences and original terrier scores from terrier result file
        # terrier_paragraph_score is file name to ranking. get reciprocal as the terrier mark
        terrier_paragraphs, terrier_paragraph_score = self.processTerrierResult(
            'data/terrier_data/terrier_results/result' + "_" + str(self.call_time) + "_turn" + str(
                turn_nbr + 1) + '.txt')
        # TODO: takes the longest time
        # line_embedding is a filename to embeddings dict
        line_embeddings = self.getParagraphInfos(terrier_paragraphs)

        # calculate terrier_score which is 1 / terrier rank
        for i in terrier_paragraph_score.keys():
            terrier_paragraph_score[i] = 1 / int(terrier_paragraph_score[i])

        # from app import query
        # if len(query) != len(conv_query_embeddings):
        #     print('attention: Line 262', 'queryLength', len(query), 'convLength', len(conv_query_embeddings))

        # para_score is a filename to int dict
        max_id, para_score = self.scoring(convquery_type, conv_query_embeddings, line_embeddings)
        return max_id, para_score, terrier_paragraphs[max_id]

    def scoring(self, convquery_type, conv_query_embeddings, line_embeddings):
        max_id = ''
        max_score = 0
        para_score = dict()
        if convquery_type == 'conv_uw':
            # current + first
            print('num', len(conv_query_embeddings))
            if len(conv_query_embeddings) == 1:
                # for each paragraph get a max line_score as para_score
                for index, embeddings in line_embeddings.items():
                    para_score[index] = 0
                    for vec in embeddings:
                        cs = cosine_similarity(conv_query_embeddings[0], [vec])
                        if cs > para_score[index]:
                            # update line_score of one paragraph
                            para_score[index] = cs
                    # for all paragraph get a max para_score as return
                    if max_score < para_score[index]:
                        max_score = para_score[index]
                        max_id = index
            else:
                # score1 + score2 is the para_score[index] value
                score1 = 0
                score2 = 0
                # for each paragraph calculate a score
                for index, embeddings in line_embeddings.items():
                    para_score[index] = 0
                    for vec in embeddings:
                        cs = cosine_similarity(conv_query_embeddings[0], [vec])
                        if cs > score1:
                            # update line_score of one paragraph
                            score1 = cs
                    for vec in embeddings:
                        cs = cosine_similarity(conv_query_embeddings[1], [vec])
                        if cs > score2:
                            # update line_score of one paragraph
                            score2 = cs
                    para_score[index] = score1 + score2
                    if max_score < para_score[index]:
                        max_score = para_score[index]
                        max_id = index
        elif convquery_type == 'conv_w1':
            score1 = 0
            score2 = 0
            score3 = 0
            # current + previous + first
            print('num', len(conv_query_embeddings))
            if len(conv_query_embeddings) == 1:
                # for each paragraph get a max line_score as para_score
                # for all paragraph get a max para_score as return
                for index, embeddings in line_embeddings.items():
                    para_score[index] = 0
                    for vec in embeddings:
                        cs = cosine_similarity(conv_query_embeddings[0], [vec])
                        # update line_score of one paragraph
                        if cs > para_score[index]:
                            para_score[index] = cs
                    # update para_score of all paragraphs
                    if para_score[index] > max_score:
                        max_score = para_score[index]
                        max_id = index
            elif len(conv_query_embeddings) == 2:
                for index, embeddings in line_embeddings.items():
                    para_score[index] = 0
                    for vec in embeddings:
                        cs = cosine_similarity(conv_query_embeddings[0], [vec])
                        # update line_score of one paragraph
                        if cs > score1:
                            score1 = cs
                    for vec in embeddings:
                        cs = cosine_similarity(conv_query_embeddings[1], [vec])
                        if cs > score2:
                            score2 = cs
                    para_score[index] = score1 + score2
                    if para_score[index] > max_score:
                        max_score = para_score[index]
                        max_id = index
            else:
                for index, embeddings in line_embeddings.items():
                    para_score[index] = 0
                    for vec in embeddings:
                        cs = cosine_similarity(conv_query_embeddings[0], [vec])
                        if cs > score1:
                            score1 = cs
                    for vec in embeddings:
                        cs = cosine_similarity(conv_query_embeddings[1], [vec])
                        if cs > score2:
                            score2 = cs
                    for vec in embeddings:
                        cs = cosine_similarity(conv_query_embeddings[2], [vec])
                        if cs > score3:
                            score3 = cs
                    # weight is (len - 1)/ len
                    weight = (len(conv_query_embeddings) - 1) / len(conv_query_embeddings)
                    para_score[index] = score1 + score2 * weight + score3
                    if para_score[index] > max_score:
                        max_score = para_score[index]
                        max_id = index

        else:
            score = 0
            final_score = 0
            print('num', len(conv_query_embeddings))
            length = len(conv_query_embeddings)
            for index, embeddings in line_embeddings.items():
                # traverse all query and get a final score for each para
                for i in range(length):
                    for vec in embeddings:
                        cs = cosine_similarity(conv_query_embeddings[i], [vec])
                        if score < cs:
                            # update para_score of all paragraphs
                            score = cs
                    weight = (i + 1) / length
                    final_score += weight * score
                para_score[index] = final_score
                if max_score < para_score[index]:
                    max_score = para_score[index]
                    max_id = index
        return max_id, para_score
