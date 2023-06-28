import networkx as nx
import json
import pickle as pkl

import numpy as np


def build_label(graph, m):
    label = 0
    ground_truth = [[0] * m] * m
    # print(graph)
    for i in range(m):
        # print(graph[i])
        for j in range(i + 1, m):
            for k in range(j + 1, m):
                # print(graph[i, j], graph[i, k], graph[j, k])
                if graph[i, j] == 1 and graph[i, k] == 1 and graph[j, k] == 1:
                    label += 1
                    ground_truth[i][j] = 1
                    ground_truth[j][i] = 1
                    ground_truth[i][k] = 1
                    ground_truth[k][i] = 1
                    ground_truth[j][k] = 1
                    ground_truth[k][j] = 1
    return label, ground_truth


def create_triangles_dataset(number, m=30, p=0.2):
    datas = []
    adjs = []
    nfs = []
    labels = []
    gts = []

    min_label = 99999999999
    max_label = -1
    for i in range(number):
        G = nx.erdos_renyi_graph(m, p, seed=i)
        adj = nx.adjacency_matrix(G).todense()
        label, ground_truth = build_label(adj, m)
        node_features = [[0.1] * 10] * m
        adjs.append(adj)
        nfs.append(node_features)
        labels.append(label)
        gts.append(ground_truth)
        if label > max_label:
            max_label = label
        if label < min_label:
            min_label = label
    print(min_label, max_label)
    graphs = np.asarray(adjs, dtype=np.float32)
    features = np.asarray(nfs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    ground_truths = np.asarray(gts, dtype=np.int32)
    # print(graphs[0], features[0], labels[0])
    path = '/Users/jiaxingzhang/Desktop/Lab/XAI_GNN/XAIG_REG/data/dataset/triangles_small.pkl'
    with open(path, 'wb') as f:
        pkl.dump((graphs, features, labels, ground_truths), f)
    pass


if __name__ == '__main__':
    create_triangles_dataset(1000)
