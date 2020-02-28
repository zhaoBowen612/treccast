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


@app.route('/getanswer', methods=['GET', 'POST'])
def getanswer():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        # content = request.get_json()
        query = [request.form.get('q1')]
        content = {"questions": query,
                   "indriRetNbr": 100,
                   "edgeThreshold": 0.5,
                   "nodeThreshold": 0.5,
                   "retNbr": 5,
                   "convquery": "conv_uw",
                   "h1": 0.5,
                   "h2": 0.25,
                   "h3": 0.25
                   }
        # print('succeed')
        # logging.info("Received Parameters: %s", content)
        # create a new CROWN instance, the model is downloaded from Google,
        # G and dict is the generated graph and dic before
        # t = CROWN(model, G, prox3_dict)
        t = Treccast(model)

        # get answer from crown
        result_paragraphs, result_ids, result_node_map, result_edge_map = t.retrieveAnswer(content)
        result = jsonify(paragraphs=result_paragraphs, ids=result_ids, nodes=result_node_map, edges=result_edge_map)
        # logging.info("TRECCAST CROWN Result: %s", result)
        # print(content)
        return render_template('index.html', result1=result_paragraphs, result2=result_ids, result3=result_node_map,
                               result4=result_edge_map)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    # logging.info("Application is starting ...")
    # logging.info("app has IP: %s and port: %i", HOST, PORT)
    app.run(host=HOST, port=PORT, threaded=True, debug=True)

'''
-questions ["What flowering plants work for cold climates?","How much cold can pansies tolerate?"]
'''
# server side "bert-serving-start -model_dir wwm_cased_L-24_H-1024_A-16/ -num_worker=1"
