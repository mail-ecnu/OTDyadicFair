import sys
import stellargraph as sg
import matplotlib.pyplot as plt
from math import isclose
import sklearn
from sklearn.decomposition import PCA
import os
import networkx as nx
import numpy as np
import pandas as pd
from stellargraph import StellarGraph, datasets
from stellargraph.data import EdgeSplitter
from collections import Counter
import multiprocessing
from IPython.display import display, HTML
from sklearn.model_selection import train_test_split
from src.main import *
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

import dill
import optuna
# is_dill = True
# if is_dill:
#     dill.load_session('./cora/beforeRB.pkl')

p = 1.0
q = 1.0
dimensions = 128
num_walks = 10
walk_length = 80
window_size = 10
num_iter = 1

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--reg1', type=float,
                    help='sym reg')
parser.add_argument('--reg2', type=float,
                    help='sym reg')
args = parser.parse_args()
reg1 = args.reg1
reg2 = args.reg2

workers = multiprocessing.cpu_count()
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec
dill.load_session(f'./cora/node_adj_{reg1}_{reg2}.pkl')

drop_weight = 0.45
emd_weight = 0.19


def RB(get_embedding, feat, name, kfold=5):
    embeddings = []
    s = []
    for i in range(len(feat.values())):
        embeddings.append(get_embedding(i))
        s.append(str(feat[i]))

    X = embeddings
    y = s
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1121218
    )
    from sklearn.svm import SVC
    clf = OneVsRestClassifier(SVC(probability=True))
    # clf = LogisticRegression(solver='lbfgs')
    clf.fit(X_train, y_train)

    y_pred_probs = clf.predict_proba(X_test)

# Calculate ROC_AUC
    results_lap = roc_auc_score(
        y_test, y_pred_probs, multi_class="ovr", average="weighted"
    )

    predict_lables = clf.predict(X_test)
    print(f"RB on the {name} graph  is:  {results_lap.mean()}")
    acc = accuracy_score(y_test, predict_lables)
    return results_lap, acc


def ERB(get_embedding, feat, name, kfold=5):
    embeddings = []
    s = []
    for i in range(len(feat.values())):
        embeddings.append(get_embedding(i))
        s.append(str(feat[i]))

    X = embeddings
    y = s
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1121218
    )
    from sklearn.svm import SVC
    clf = OneVsRestClassifier(SVC(probability=True))
    # clf = LogisticRegression(solver='lbfgs')
    clf.fit(X_train, y_train)

    y_pred_probs = clf.predict_proba(X_test)

# Calculate ROC_AUC
    results_lap = roc_auc_score(
        y_test, y_pred_probs, multi_class="ovr", average="weighted"
    )
    y_predict = clf.predict(X_test)
    acc = accuracy_score(y_test, y_predict)
    print(f"RB on the {name} graph  is:  {results_lap.mean()}")
    return results_lap, acc

def vis_pca(name, best_result, examples_test, embedding_test):
    link_features = link_examples_to_features(
        examples_test, embedding_test, best_result["binary_operator"]
    )

    # Learn a projection from 128 dimensions to 2
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(link_features)

    # plot the 2-dimensional points
    plt.figure(figsize=(16, 12))
    plt.scatter(
        X_transformed[:, 0],
        X_transformed[:, 1],
        c=np.where(labels_test == 1, "b", "r"),
        alpha=0.5,
    )
    plt.tight_layout()
    plt.savefig(f'figs/dblp/{name}-pca.jpg')

def vis_nx(name, g):
    # Visualisation of the generated graph

    #Retrieve indexes of node in each group
    s = nx.get_node_attributes(g, 's')
    idx_ps = []
    labels = list(set(s.values()))
    for val in labels:
        idx_ps.append(get_keys_from_value(s, val))


    # Draw the graph
    pos = nx.spring_layout(g)
    i = 0
    colors = ['steelblue', 'gold', 'green', 'red', 'orange']
    for idx_p in idx_ps:
        nx.draw_networkx_nodes(g, pos=pos, node_size=0.1, nodelist=idx_p, node_color=colors[i], label=f'S = {labels[i]}')

    nx.draw_networkx_edges(g, pos=pos)
    plt.legend(loc="upper left", scatterpoints=1, prop={'size': 15})
    plt.tight_layout()
    plt.savefig(f'figs/dblp/{name}-nx.jpg')

def node2vec_embedding(graph, name):
    rw = BiasedRandomWalk(graph)
    walks = rw.run(graph.nodes(), n=num_walks, length=walk_length, p=p, q=q)
    print(f"Number of random walks for '{name}': {len(walks)}")

    model = Word2Vec(
        walks,
        vector_size=dimensions,
        window=window_size,
        min_count=0,
        sg=1,
        workers=workers,
#         iter=num_iter,
    )

    def get_embedding(u):
        return model.wv[u]

    return get_embedding


# 1. link embeddings
def link_examples_to_features(link_examples, transform_node, binary_operator):
    return [
        binary_operator(transform_node(src), transform_node(dst))
        for src, dst in link_examples
    ]


# 2. training classifier
def train_link_prediction_model(
    link_examples, link_labels, get_embedding, binary_operator
):
    clf = link_prediction_classifier()
    link_features = link_examples_to_features(
        link_examples, get_embedding, binary_operator
    )
    clf.fit(link_features, link_labels)
    return clf


def link_prediction_classifier(max_iter=3000):
    lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
    return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])


# 3. and 4. evaluate classifier
def evaluate_link_prediction_model(
    clf, link_examples_test, link_labels_test, get_embedding, binary_operator
):
    link_features_test = link_examples_to_features(
        link_examples_test, get_embedding, binary_operator
    )
    score, acc = evaluate_roc_auc(clf, link_features_test, link_labels_test)
    return score, acc



def evaluate_roc_auc(clf, link_features, link_labels):
    predicted = clf.predict_proba(link_features)

    # check which class corresponds to positive links
    positive_column = list(clf.classes_).index(1)
    predicted_labels = clf.predict(link_features)
    return roc_auc_score(link_labels, predicted[:, positive_column]), accuracy_score(link_labels, predicted_labels)

def operator_hadamard(u, v):
    return u * v


def operator_l1(u, v):
    return np.abs(u - v)


def operator_l2(u, v):
    return (u - v) ** 2


def operator_avg(u, v):
    return (u + v) / 2.0


def run_link_prediction(embedding_train, binary_operator, examples_train, labels_train, examples_model_selection, labels_model_selection,):
    clf = train_link_prediction_model(
        examples_train, labels_train, embedding_train, binary_operator
    )
    score, acc = evaluate_link_prediction_model(
        clf,
        examples_model_selection,
        labels_model_selection,
        embedding_train,
        binary_operator,
    )

    return {
        "classifier": clf,
        "binary_operator": binary_operator,
        "score": score,
        'acc': acc
    }

def AUC_print(results):
    print(pd.DataFrame(
    [(result["binary_operator"].__name__, result["score"]) for result in results],
    columns=("name", "ROC AUC score"),
    ).set_index("name"))

def DI(best_result, examples_test, g, embedding_test):
    link_features_test = link_examples_to_features(
        examples_test, embedding_test, best_result["binary_operator"]
    )
    xor0, xor1 = [], []
    feat = nx.get_node_attributes(g, 's')

    for i in range(len(examples_test)):
        if feat[examples_test[i][0]] == feat[examples_test[i][1]]:
            xor0.append(link_features_test[i])
        else:
            xor1.append(link_features_test[i])
    y0 = best_result["classifier"].predict(xor0)
    score0 = sklearn.metrics.accuracy_score(y0, np.ones_like(y0))
    y1 = best_result["classifier"].predict(xor1)
    score1 = sklearn.metrics.accuracy_score(y1, np.ones_like(y1))
    return score1/score0

binary_operators = [operator_hadamard, operator_l1, operator_l2, operator_avg]

def parse_cora(plot=False):
    path = "./data/cora/"

    id2index = {}

    label2index = {
        'Case_Based': 0,
        'Genetic_Algorithms': 1,
        'Neural_Networks': 2,
        'Probabilistic_Methods': 3,
        'Reinforcement_Learning': 4,
        'Rule_Learning': 5,
        'Theory': 6
    }

    features = []
    labels = []

    with open(path + 'cora.content', 'r') as f:
        i = 0
        for line in f.readlines():
            items = line.strip().split('\t')

            id = items[0]

            # 1-hot encode labels
            label = np.zeros(len(label2index))
            label[label2index[items[-1]]] = 1
            labels.append(items[-1])

            # parse features
            features.append([int(x) for x in items[1:-1]])

            id2index[id] = i
            i += 1

    features = np.asarray(features, dtype='float32')
    labels = np.array(labels)
    # labels = np.asarray(labels, dtype='int32')

    n_papers = len(id2index)

    adj = np.zeros((n_papers, n_papers), dtype='float32')

    with open(path + 'cora.cites', 'r') as f:
        for line in f.readlines():
            items = line.strip().split('\t')
            adj[ id2index[items[0]], id2index[items[1]] ] = 1.0
            # undirected
            adj[ id2index[items[1]], id2index[items[0]] ] = 1.0

    G = nx.from_numpy_matrix(adj, nx.Graph())
    feat_dict, label_dict = {}, {}
    for i in range(features.shape[0]):
        feat_dict[i] = features[i]
        label_dict[i] = labels[i]
    nx.set_node_attributes(G, feat_dict, name='feat')
    nx.set_node_attributes(G, label_dict, name='s')
    return G


def load_data():
    G = parse_cora()
    # G = nx.relabel.convert_node_labels_to_integers(G, first_label=0, ordering='default')
    return G

def emd_repair(graph_train, num_iter=1e6, edge_weight=0.2):
    emd_adj, s_emd, gamma, M = multi_total_repair(graph_train,  num_iter=num_iter, metric='euclidean', log=False)
    print('emd edges', np.sum(np.array(emd_adj) >= edge_weight))
    emd_g = nx.from_numpy_matrix(emd_adj)
    # Filter out the smallest weights to keep a reasonable density
    list_edge = [(u, v) for (u, v, d) in emd_g.edges(data=True) if d['weight'] < edge_weight]
    emd_g.remove_edges_from(list_edge)
    nx.set_node_attributes(emd_g, nx.get_node_attributes(graph_train, 's'), name='s')
    print('Assortativity coeffcient on the emd graph: %0.3f'
      % nx.attribute_assortativity_coefficient(emd_g, 's'))
    return emd_g

def drop_repair(graph_train, edge_weight=0.2):
    sens = nx.get_node_attributes(graph_train, 's')
    sens_ls = []
    for i in range(len(sens)):
        sens_ls.append(sens[i])
    sens_ls = np.array(sens_ls)
    mij = np.random.rand(len(sens_ls)*len(sens_ls)).reshape(len(sens_ls), len(sens_ls))
    for i in range(0, len(sens_ls)):
        # import pdb; pdb.set_trace()
        mij[i][sens_ls == sens_ls[i]] = 0
        mij[i][sens_ls != sens_ls[i]] = 1
        myrand = np.random.rand(len(sens_ls))
        mij[i][myrand < 0.5 - edge_weight] = 1- mij[i][myrand < 0.5 - edge_weight]
    drop_adj = nx.adjacency_matrix(graph_train) * mij
    drop_g = nx.from_numpy_matrix(drop_adj)
    # Filter out the smallest weights to keep a reasonable density
    nx.set_node_attributes(drop_g, nx.get_node_attributes(graph_train, 's'), name='s')
    print('Assortativity coeffcient on the drop graph: %0.3f'
      % nx.attribute_assortativity_coefficient(drop_g, 's'))
    return drop_g


def sym_repair_adj(graph_train, num_iter=1e6, reg=1e-9):
    emd_adj, emd_nodes, s_emd, gamma, M = multi_node_sym_total_repair(graph_train,  num_iter=num_iter, metric='euclidean', log=False, reg=reg, reg1=reg1, reg2=reg2)

    return emd_adj, emd_nodes

def node_repair_adj(graph_train, num_iter=1e6, reg=1e-9):
    emd_adj, emd_nodes, s_emd, gamma, M = multi_node_total_repair(graph_train,  num_iter=num_iter, metric='euclidean', log=False, reg=reg, reg1=reg1, reg2=reg2)

    return emd_adj, emd_nodes

def node_repair(graph_train, emd_adj, emd_nodes, edge_weight=0.2):
    print('emd edges', np.sum(np.array(emd_adj) >= edge_weight))
    emd_g = nx.from_numpy_matrix(emd_adj)
    # Filter out the smallest weights to keep a reasonable density
    list_edge = [(u, v) for (u, v, d) in emd_g.edges(data=True) if d['weight'] < edge_weight]
    emd_g.remove_edges_from(list_edge)

    emd_nodes_dict = {}
    for i in range(emd_nodes.shape[0]):
        emd_nodes_dict[i] = emd_nodes[i]
    nx.set_node_attributes(emd_g, emd_nodes_dict, name='feat')
    # import pdb; pdb.set_trace()
    nx.set_node_attributes(emd_g, nx.get_node_attributes(graph_train, 's'),name='s')
    print('Assortativity coeffcient on the node repair graph: %0.3f'
      % nx.attribute_assortativity_coefficient(emd_g, 's'))
    return emd_g


def sym_repair(graph_train, emd_adj, emd_nodes, edge_weight=0.2):
    print('emd edges', np.sum(np.array(emd_adj) >= edge_weight))
    emd_g = nx.from_numpy_matrix(emd_adj)
    # Filter out the smallest weights to keep a reasonable density
    list_edge = [(u, v) for (u, v, d) in emd_g.edges(data=True) if d['weight'] < edge_weight]
    emd_g.remove_edges_from(list_edge)

    emd_nodes_dict = {}
    for i in range(emd_nodes.shape[0]):
        emd_nodes_dict[i] = emd_nodes[i]
    nx.set_node_attributes(emd_g, emd_nodes_dict, name='feat')
    # import pdb; pdb.set_trace()
    nx.set_node_attributes(emd_g, nx.get_node_attributes(graph_train, 's'),name='s')
    print('Assortativity coeffcient on the emd graph: %0.3f'
      % nx.attribute_assortativity_coefficient(emd_g, 's'))
    return emd_g

def AUC_test(best_result, examples_test,labels_test, embedding_test):

    test_score = evaluate_link_prediction_model(
        best_result["classifier"],
        examples_test,
        labels_test,
        embedding_test,
        best_result["binary_operator"],
    )
    print(
        f"ROC AUC score on test set using '{best_result['binary_operator'].__name__}': {test_score}"
    )


def edge_pred(examples_train, examples_model_selection, labels_train, labels_model_selection, embedding_train):
    results = [run_link_prediction(embedding_train, op, examples_train, labels_train, examples_model_selection,
            labels_model_selection,) for op in binary_operators]
    best_result = max(results, key=lambda result: result["score"])
    AUC_print(results)
    return best_result

def get_xor_label(examples_train, graph_train):
    feat = nx.get_node_attributes(graph_train, 's')
    xor_labels = []
    for example in examples_train:
        if feat[example[0]] == feat[example[1]]:
            xor_labels.append(0)
        else:
            xor_labels.append(1)
    return xor_labels

is_dill = True
if not is_dill:
    G = load_data()

    # nx.set_node_attributes(G, node_feat, 's')

    print('Assortativity coeffcient on the origin graph: %0.3f'
        % nx.attribute_assortativity_coefficient(G, 's'))
    print("Correcting the graph with EMD")
    edge_splitter_test = EdgeSplitter(G)
    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from graph, and obtain the
    # reduced graph graph_test with the sampled links removed:
    graph_test, examples_test, labels_test = edge_splitter_test.train_test_split(
        p=0.1, method="global"
    )

    # Do the same process to compute a training subset from within the test graph

    edge_splitter_train = EdgeSplitter(graph_test, G)
    graph_train, examples, labels = edge_splitter_train.train_test_split(
        p=0.1, method="global"
    )

    # REPAIRG
    drop_g = drop_repair(graph_train, edge_weight=drop_weight)
    emd_g = emd_repair(graph_train, num_iter=1e6, edge_weight=0.12)

    # EMBEDDING with node2vec
    embedding_train = node2vec_embedding(StellarGraph.from_networkx(graph_train, node_features='feat'), "Train Graph")
    emd_embedding_train = node2vec_embedding(StellarGraph.from_networkx(emd_g, node_features='feat'), "EMD Train Graph")
    drop_embedding_train = node2vec_embedding(StellarGraph.from_networkx(drop_g), "Drop Train Graph")

    # TRAIN and VAL
    (
        examples_train,
        examples_model_selection,
        labels_train,
        labels_model_selection,
    ) = train_test_split(examples, labels, train_size=0.75, test_size=0.25)

    print('origin')
    ori_best_result = edge_pred(examples_train, examples_model_selection, labels_train, labels_model_selection, embedding_train)

    print('drop')
    drop_best_result = edge_pred(examples_train, examples_model_selection, labels_train, labels_model_selection, drop_embedding_train)

    print('emd')
    emd_best_result = edge_pred(examples_train, examples_model_selection, labels_train, labels_model_selection, emd_embedding_train)

    embedding_test = node2vec_embedding(StellarGraph.from_networkx(graph_test), "Test Graph")
    print('origin graph')
    AUC_test(ori_best_result, examples_test,labels_test, embedding_test)

    print('drop graph')
    drop_graph_test = drop_repair(graph_test, edge_weight=drop_weight)
    drop_embedding_test = node2vec_embedding(StellarGraph.from_networkx(drop_graph_test), "Drop Test Graph")
    AUC_test(drop_best_result, examples_test,labels_test, drop_embedding_test)

    print('emd graph')
    emd_graph_test = emd_repair(graph_test, num_iter=1e6, edge_weight=emd_weight)
    emd_embedding_test = node2vec_embedding(StellarGraph.from_networkx(emd_graph_test), "EMD Test Graph")
    AUC_test(emd_best_result, examples_test,labels_test, emd_embedding_test)

    # RB
    ori_feat = nx.get_node_attributes(graph_train, 's')
    RB(embedding_train, ori_feat, 'origin', kfold=5)
    drop_feat = nx.get_node_attributes(drop_g, 's')
    RB(drop_embedding_train, drop_feat, 'emd', kfold=5)
    emd_feat = nx.get_node_attributes(emd_g, 's')
    RB(emd_embedding_train, emd_feat, 'emd', kfold=5)

    # DI

    ori_di = DI(ori_best_result, examples_test, graph_test, embedding_test)
    drop_di = DI(drop_best_result, examples_test, graph_test, drop_embedding_test)
    emd_di = DI(emd_best_result, examples_test, graph_test, emd_embedding_test)
    print(f'ori_DI: {ori_di}, drop_DI: {drop_di}, emd_DI: {emd_di}')

    dill.dump_session('./cora/beforeERB.pkl')
    # ERB
    print('Edge RB')
    xor_train_label, xor_val_label = get_xor_label(examples_train, graph_train), get_xor_label(examples_model_selection, graph_train)
    print('ori')
    edge_pred(examples_train, examples_model_selection, xor_train_label, xor_val_label, embedding_train)
    print('drop')
    edge_pred(examples_train, examples_model_selection, xor_train_label, xor_val_label, drop_embedding_train)
    print('emd')
    edge_pred(examples_train, examples_model_selection, xor_train_label, xor_val_label, emd_embedding_train)

    # VISUALIZATION
    # Calculate edge features for test data
    vis_pca('origin', ori_best_result, examples_test, embedding_test)
    vis_pca('drop', drop_best_result, examples_test, drop_embedding_test)
    vis_pca('emd', emd_best_result, examples_test, emd_embedding_test)
    vis_nx('origin', graph_train)
    vis_nx('drop', drop_g)
    vis_nx('emd', emd_g)
    dill.dump_session('./cora/beforeSym.pkl')
    exit()
# else:
#     dill.load_session(f'./cora/sym_adj_{reg}.pkl')
    # reg = 1e-3
    # sym_g = sym_repair_adj(graph_train, num_iter=1e4, reg=reg)
    # sym_testg = sym_repair_adj(graph_test, num_iter=1e4, reg=reg)

    # results = [run_link_prediction(embedding_train, op, examples_train, labels_train, examples_model_selection,
    #         labels_model_selection,) for op in binary_operators]
    # best_result = max(results, key=lambda result: result["score"])
    # AUC_print(results)

    # emd_results = [run_link_prediction(emd_embedding_train, op, examples_train, labels_train, examples_model_selection,
    #         labels_model_selection,) for op in binary_operators]

    # emd_best_result = max(emd_results, key=lambda result: result["score"])

    # print('emd repair')
    # AUC_print(emd_results)

    # # TEST AUC
    # embedding_test = node2vec_embedding(StellarGraph.from_networkx(graph_test, node_features='feat'), "Test Graph")
    # print('origin graph')
    # AUC_test(best_result, examples_test,labels_test, embedding_test)

    # print('emd graph')
    # emd_graph_test = emd_repair(graph_test)
    # emd_embedding_test = node2vec_embedding(StellarGraph.from_networkx(emd_graph_test, node_features='feat'), "EMD Test Graph")
    # AUC_test(emd_best_result, examples_test,labels_test, emd_embedding_test)

    # ori_feat = nx.get_node_attributes(graph_test, 's')
    # RB(embedding_test, ori_feat, 'origin', kfold=5)
    # emd_feat = nx.get_node_attributes(emd_graph_test, 's')
    # RB(emd_embedding_test, emd_feat, 'emd', kfold=5)

    # # DI


    # ori_di = DI(best_result, examples_test, graph_test, embedding_test)
    # emd_di = DI(emd_best_result, examples_test, graph_test, emd_embedding_test)
    # print(f'ori_DI: {ori_di}, emd_DI: {emd_di}')
    # # VISUALIZATION
    # # Calculate edge features for test data
    # vis_pca('origin', best_result, examples_test, embedding_test)
    # vis_pca('emd', emd_best_result, examples_test, embedding_test)
    # vis_nx('origin', graph_train)
    # vis_nx('emd', emd_g)

    # dill.dump_session('./cora/beforeRB.pkl')
    # reg = 1e-3
    # sym_g = sym_repair_adj(graph_train, num_iter=1e4, reg=reg)
    # sym_testg = sym_repair_adj(graph_test, num_iter=1e4, reg=reg)



# multi_adj, multi_nodes = node_repair_adj(graph_train, num_iter=1e4, reg=0)
# sym_test_adj, sym_test_nodes = node_repair_adj(graph_test, num_iter=1e4, reg=0)
# dill.dump_session(f'./cora/node_adj_{reg1}_{reg2}.pkl')

def objective(trial):

    edge_weight = trial.suggest_float("lr", 0.12, 0.2, log=True)

    AUCs = []
    ACCs = []
    RBs = []
    RB_ACCs = []
    DIs = []
    ERBs = []
    ERB_ACCs = []
    sym_g = node_repair(graph_train, multi_adj, multi_nodes, edge_weight=edge_weight)

# for i in range(5):

    sym_embedding_train = node2vec_embedding(StellarGraph.from_networkx(sym_g, node_features='feat'), "Train Sym Graph")
    sym_best_result = edge_pred(examples_train, examples_model_selection, labels_train, labels_model_selection, sym_embedding_train)
    test_auc = sym_best_result['score']
    test_acc = sym_best_result['acc']
    AUCs.append(test_auc)
    ACCs.append(test_acc)


    sym_feat = nx.get_node_attributes(sym_g, 's')
    rb_auc, rb_acc = RB(sym_embedding_train, sym_feat, 'sym', kfold=5)
    RBs.append(rb_auc)
    RB_ACCs.append(rb_acc)

    sym_di = DI(sym_best_result, examples_train, sym_g, sym_embedding_train)
    DIs.append(sym_di)
    print(f'sym_DI: {sym_di}')
    print('Edge RB')
    xor_train_label, xor_val_label = get_xor_label(examples_train, graph_train), get_xor_label(examples_model_selection, graph_train)
    print('sym')
    best_ERB_result = edge_pred(examples_train, examples_model_selection, xor_train_label, xor_val_label, sym_embedding_train)
    erb_auc, erb_acc = best_ERB_result['score'], best_ERB_result['acc']
    ERBs.append(erb_auc)
    ERB_ACCs.append(erb_acc)
    return ACCs[0], abs(ERB_ACCs[0]-0.5)

if __name__ == '__main__':

    study_name = 'fairnode'
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.create_study(study_name=study_name, storage=storage_name, directions=["maximize", "minimize"], load_if_exists=True)

    study.optimize(lambda trial: objective(trial), n_trials=50)


    print("Best trial:")
    trial = study.best_trials

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
# mydict = {'AUCs': np.array(AUCs),
#         'ACCs': np.array(ACCs),
#         'RBs': np.array(RBs),
#         'RB_ACCs': np.array(RB_ACCs),
#         'DIs': np.array(DIs),
#         'ERBs': np.array(ERBs),
#         'ERB_ACCs': np.array(ERB_ACCs)
#         }

# for name in mydict.keys():
#     print(f'{name}: {mydict[name].mean()}, {np.sqrt(mydict[name].var())} \n')
