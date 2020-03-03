from flask import Flask, request, jsonify, render_template, redirect
# import os
import pickle
import gensim
# import json
import networkx as nx
from flask_cors import CORS
import logging
from bert_serving.client import BertClient
from treccast import *

# logging.basicConfig(filename="logging/flask_demo.log", level=logging.INFO)

# add name of host and port here
HOST = '127.0.0.1'
PORT = 5000

# load word embeddings
model = BertClient()
# logging.info("embeddings loaded")

# load proximity information
# with open('data/graph_data/prox_3.pickle', 'rb') as handle:
#     prox3_dict = pickle.load(handle)  # pickle is used for serialization
# logging.info("proximity infos loaded")
# print("load proximity information")

# load graph created from marco corpus
# G = nx.read_gpickle("data/graph_data/marco_graph_pmi3_edges.gpickle")
# logging.info("Graph successfully loaded")
# print("load graph created from marco corpus")


app = Flask(__name__)
CORS(app)

# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
query = []
result = []


@app.route('/getanswer', methods=['GET', 'POST'])
def getanswer():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        # content = request.get_json()
        query.append(request.form.get('q1'))
        content = {"questions": query,
                   "indriRetNbr": request.form.get('inum'),
                   "retNbr": request.form.get('rnum'),
                   "convquery": request.form.get('sel'),
                   "h1": request.form.get('h1'),
                   "h2": request.form.get('h2'),
                   }
        t = Treccast(model)
        result_ids, para_score, result_content = t.retrieveAnswer(content)
        # 评估的时候需要返回所有段落及评分
        result.append((result_ids, para_score[result_ids], result_content))
        # result = jsonify(paragraphs=result_paragraphs, ids=result_ids, nodes=result_node_map, edges=result_edge_map)
        return render_template('index.html', result=result)


@app.route('/')
def index():
    query.clear()
    result.clear()
    return render_template('index.html')


if __name__ == '__main__':
    # logging.info("Application is starting ...")
    # logging.info("app has IP: %s and port: %i", HOST, PORT)
    app.run(host=HOST, port=PORT, threaded=True, debug=True)

# server side "bert-serving-start -model_dir wwm_cased_L-24_H-1024_A-16/ -num_worker=1"
