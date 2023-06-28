import json
import pickle as pkl
import numpy as np
import os
from numpy.random.mtrand import RandomState
from torch_geometric.data import Data
import torch
from utils.dataset_util.data_utils import preprocess_features, preprocess_adj, adj_to_edge_index, load_real_dataset
from utils.plot_graph import plot_graph, plot_expl


class DatasetLoader:
    def __init__(self, dataset_name, input_type, task_type):
        self.dataset_path = '/Users/jiaxingzhang/Desktop/Lab/XAI_GNN/XAIG_REG/data/dataset/'
        self.data_root_path = '/Users/jiaxingzhang/Desktop/Lab/XAI_GNN/XAIG_REG/data/'
        self.dataset_name = dataset_name
        self.input_type = input_type
        self.type = task_type  # reg or cls
        self.indices = []
        self.node_indices = []  # mark the central node of the k-hop graph
        self.graphs = []
        self.features = []
        self.labels = []
        self.node_ground_truth = []
        self.edge_ground_truth = []
        self.train_mask = []
        self.val_mask = []
        self.test_mask = []
        self.train_data_list = []
        self.val_data_list = []
        self.test_data_list = []
        pass

    def load_dataset(self, skip_preproccessing=False, shuffle=True):
        """High level function which loads the dataset
        by calling others spesifying in nodes or graphs.

        Keyword arguments:
        :param _dataset: Which dataset to load. Choose from "syn1", "syn2", "syn3", "syn4", "ba2" or "mutag"
        :param skip_preproccessing: Whether or not to convert the adjacency matrix to an edge matrix.
        :param shuffle: Should the returned dataset be shuffled or not.
        :returns: multiple np.arrays
        """
        print(f"Loading {self.dataset_name} dataset")
        if self.dataset_name[:3] == "syn":  # Load node_dataset
            adj, features, labels, train_mask, val_mask, test_mask = self._load_node_dataset()
            features = preprocess_features(features).astype('float32')
            if skip_preproccessing:
                graph = adj
            else:
                graph = preprocess_adj(adj)[0].astype('int64').T
            labels = np.argmax(labels, axis=1)
            # return graph, preprocessed_features, labels, train_mask, val_mask, test_mask
        else:  # Load graph dataset
            graph, features, labels, train_mask, val_mask, test_mask, node_ground_truth, edge_ground_truth = self.load_graph_dataset(shuffle)
            self.node_ground_truth = node_ground_truth
            self.edge_ground_truth = edge_ground_truth

        self.indices = [i for i in range(len(graph))]
        self.graphs = graph
        self.features = features
        self.labels = labels
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

    def _load_node_dataset(self):
        """Load a node dataset.

        :param _dataset: Which dataset to load. Choose from "syn1", "syn2", "syn3" or "syn4"
        :returns: np.array
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + '/pkls/' + self.dataset_name + '.pkl'
        with open(path, 'rb') as fin:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = pkl.load(fin)
        labels = y_train
        labels[val_mask] = y_val[val_mask]
        labels[test_mask] = y_test[test_mask]

        return adj, features, labels, train_mask, val_mask, test_mask

    def load_graph_dataset(self, shuffle=True):
        """Load a graph dataset and optionally shuffle it.
        :param shuffle: Boolean. Whether to shuffle the loaded dataset.
        :returns: np.array
        """

        # Load the chosen dataset from the pickle file.
        if self.dataset_name == "mutag":
            dir_path = os.path.dirname(os.path.realpath(__file__))
            path = dir_path + '/dataset/' + "Mutagenicity" + '.pkl'
            if not os.path.exists(path):  # pkl not yet created
                print("Mutag dataset pickle is not yet created, doing this now. Can take some time")
                adjs, features, labels = load_real_dataset(path, dir_path + '/Mutagenicity/Mutagenicity_')
                print("Done with creating the mutag dataset")
            else:
                with open(path, 'rb') as fin:
                    adjs, features, labels = pkl.load(fin)
        elif self.dataset_name in ['bareg1', 'bareg2']:
            path = os.path.join(self.dataset_path, self.dataset_name + '.pkl')
            with open(path, 'rb') as fin:
                adjs, features, labels, ground_truth = pkl.load(fin)
                edge_ground_truth, node_ground_truth = self.load_bareg_dataset_ground_truth(adjs, ground_truth)
        elif self.dataset_name in ['triangles', 'triangles_small']:
            path = os.path.join(self.dataset_path, self.dataset_name + '.pkl')
            with open(path, 'rb') as fin:
                adjs, features, labels, ground_truth = pkl.load(fin)
                edge_ground_truth, node_ground_truth = self.load_triangles_dataset_ground_truth(adjs, ground_truth)
        elif self.dataset_name in ['ba2motif']:
            path = os.path.join(self.dataset_path, self.dataset_name + '.pkl')
            adjs, features, labels, edge_ground_truth, node_ground_truth = self.load_ba2_ground_truth(path)
        elif self.dataset_name in ['crippen']:
            return self.load_crippen()
        else:
            print("Unknown dataset")
            raise NotImplemented

        n_graphs = adjs.shape[0]
        indices = np.arange(0, n_graphs)
        if shuffle:
            prng = RandomState(
                42)  # Make sure that the permutation is always the same, even if we set the seed different
            indices = prng.permutation(indices)

        # Create shuffled data
        adjs = adjs[indices]
        features = features[indices].astype('float32')
        labels = labels[indices]
        node_ground_truth = node_ground_truth[indices]
        edge_ground_truth = edge_ground_truth[indices]

        # Create masks
        train_indices = np.arange(0, int(n_graphs * 0.8))
        val_indices = np.arange(int(n_graphs * 0.8), int(n_graphs * 0.9))
        test_indices = np.arange(int(n_graphs * 0.9), n_graphs)
        train_mask = np.full(n_graphs, False, dtype=bool)
        train_mask[train_indices] = True
        val_mask = np.full(n_graphs, False, dtype=bool)
        val_mask[val_indices] = True
        test_mask = np.full(n_graphs, False, dtype=bool)
        test_mask[test_indices] = True

        if self.input_type == 'sparse':
            edge_index = adj_to_edge_index(adjs)
        elif self.input_type == 'dense':
            edge_index = adjs
        else:
            assert 0
        return edge_index, features, labels, train_mask, val_mask, test_mask, node_ground_truth, edge_ground_truth

    def load_bareg_dataset_ground_truth(self, adjs, ground_truths):
        """Load the ground truth from the bareg dataset.
        :returns: np.array
        """

        n_graphs = adjs.shape[0]
        indices = np.arange(0, n_graphs)

        # Create shuffled data
        edge_indices = adj_to_edge_index(adjs)
        np_edge_labels = []
        np_node_labels = []

        # Obtain the edge labels.
        insert = 20
        skip = 5
        for i in range(len(edge_indices)):
            edge_index = edge_indices[i]
            ground_truth = ground_truths[i]
            ground_truth = [j for j in ground_truth if j != 0]
            np_node_labels.append(np.array(ground_truth))
            # print(edge_index, ground_truth)
            labels = []
            for pair in edge_index.T:
                r = pair[0]
                c = pair[1]
                # print(r, c)
                if r in ground_truth and c in ground_truth:
                    labels.append(1)
                else:
                    labels.append(0)

            # print(labels)
            # assert 0
            np_edge_labels.append(np.array(labels))

        return np.array(np_edge_labels), np.array(np_node_labels)

    def load_triangles_dataset_ground_truth(self, adjs, ground_truths):
        """Load the ground truth from the bareg dataset.
        :returns: np.array
        """

        n_graphs = adjs.shape[0]
        indices = np.arange(0, n_graphs)

        # Create shuffled data
        edge_indices = adj_to_edge_index(adjs)
        np_edge_labels = []
        np_node_labels = []

        # Obtain the edge labels.
        insert = 20
        skip = 5
        for i in range(len(edge_indices)):
            edge_index = edge_indices[i]
            ground_truth = ground_truths[i]
            node_label = []
            # print(edge_index, ground_truth)
            labels = []
            for pair in edge_index.T:
                r = pair[0]
                c = pair[1]
                # print(r, c)
                if ground_truth[r][c] == 1:
                    labels.append(1)
                    if r not in node_label:
                        node_label.append(r)
                    if c not in node_label:
                        node_label.append(c)
                else:
                    labels.append(0)

            np_node_labels.append(np.array(node_label))

            # print(labels)
            # assert 0
            np_edge_labels.append(np.array(labels))

        return np.array(np_edge_labels), np.array(np_node_labels)

    def load_ba2_ground_truth(self, path):
        """Load a the ground truth from the ba2motif dataset.

        :param path: path to data
        :returns: np.array, np.array
        """
        with open(path, 'rb') as fin:
            adjs, features, graph_labels = pkl.load(fin)

        edge_indices = adj_to_edge_index(adjs)
        np_edge_labels = []
        np_node_labels = []
        # Obtain the edge labels.
        insert = 20
        skip = 5
        for edge_index in edge_indices:
            labels = []
            for pair in edge_index.T:
                r = pair[0]
                c = pair[1]
                # In line with the original PGExplainer code we determine the ground truth based on the location in the index
                if r >= insert and r < insert + skip and c >= insert and c < insert + skip:
                    labels.append(1)
                else:
                    labels.append(0)
            np_edge_labels.append(np.array(labels))
            np_node_labels.append(np.array([20, 21, 22, 23, 24]))
        return adjs, features, graph_labels, np.array(np_edge_labels), np.array(np_node_labels)

    def load_crippen(self):
        path = os.path.join(self.dataset_path)
        train_path = self.dataset_path + "crippen_train" + '.pkl'
        test_path = self.dataset_path + "crippen_test" + '.pkl'
        with open(train_path, 'rb') as f:
            train_data = pkl.load(f)
        with open(test_path, 'rb') as f:
            test_data = pkl.load(f)
        n_graphs = len(test_data['y']) + len(train_data['y'])
        indices = np.arange(0, n_graphs)
        # No shuffle, fixed split data
        train_data['nodes_features'].extend(test_data['nodes_features'])
        node_features = train_data['nodes_features']
        train_data['edge_features'].extend(test_data['edge_features'])
        edge_features = train_data['edge_features']
        # features are one hot code not continuous
        train_data['y'].extend(test_data['y'])
        labels = train_data['y']
        # process edge attributes
        method = 'contribution-70'
        edge_attributes = []
        edge_values = []
        for idx, graph in enumerate(test_data['edges']):
            tmp_edge = []
            for index in graph:
                # value of edges
                tmp = (test_data['node_attributes'][idx][index][0] + test_data['node_attributes'][idx][index][1]) / 2
                tmp_edge.append(tmp)
            # convert to absolute value
            tmp_edge = np.abs(np.array(tmp_edge))
            # rank and stamp attributes
            if method == 'contribution-70':
                total = np.sum(tmp_edge)
                rank = np.flip(np.argsort(tmp_edge))  # from larger abs to smaller abs
                accum = np.cumsum(tmp_edge[rank]) / total  # cumulative contribution along ranked absolute value
                positive = accum < 0.7  # threshold
                edge_ground_truth = np.zeros(len(tmp_edge))
                edge_ground_truth[rank[positive]] = 1
            else:
                raise NotImplementedError
            edge_values.append(tmp_edge)
            edge_attributes.append(edge_ground_truth)
        test_data.update({'edge_attributes': edge_attributes})
        # padding for train set
        edge_attributes = []
        for idx, graph in enumerate(train_data['edges']):
            edge_ground_truth = np.zeros(len(graph))
            edge_attributes.append(edge_ground_truth)
        train_data.update({'edge_attributes': edge_attributes})

        train_data['edge_attributes'].extend(test_data['edge_attributes'])
        edge_ground_truth = train_data['edge_attributes']

        node_attributes = []
        for idx, graph in enumerate(train_data['nodes_features']):
            node_attributes.append(np.zeros(graph.shape[0]))
        train_data.update({'node_attributes': node_attributes})

        train_data['node_attributes'].extend(test_data['node_attributes'])
        node_ground_truth = train_data['node_attributes']
        # Create masks
        train_indices = np.arange(0, len(train_data['adj_matrix']))  # adj untouched
        val_indices = np.arange(len(train_data['adj_matrix']), n_graphs)
        test_indices = val_indices
        train_mask = np.full(n_graphs, False, dtype=bool)
        train_mask[train_indices] = True
        val_mask = np.full(n_graphs, False, dtype=bool)
        val_mask[val_indices] = True
        test_mask = np.full(n_graphs, False, dtype=bool)
        test_mask[test_indices] = True
        # edge untouched
        train_data['edges'].extend(test_data['edges'])
        edge_index = []
        for edges in train_data['edges']:
            source = edges[:, 0].tolist()
            target = edges[:, 1].tolist()
            edge_index.append(np.array([source, target]))

        return edge_index, node_features, labels, train_mask, val_mask, test_mask, node_ground_truth, edge_ground_truth

    def create_data_list(self):
        """
        Convert the numpy data to torch tensors and save them in a list.
        :params mask: mask, used to filter the data
        :retuns: list; contains the dataset
        """
        def create_with_mask(mask):
            indices = np.argwhere(mask).squeeze()
            data_list = []
            for i in indices:
                x = torch.tensor(self.features[i])
                edge_index = torch.tensor(self.graphs[i])

                if self.type == 'cls':
                    y = torch.tensor(self.labels[i].argmax())
                else:
                    y = torch.tensor(self.labels[i])
                edge_mask = torch.ones(edge_index.size(1))
                data = Data(x=x, edge_index=edge_index, edge_mask=edge_mask, y=y)
                data_list.append(data)
                # print(data)
            return data_list

        self.train_data_list = create_with_mask(self.train_mask)
        self.val_data_list = create_with_mask(self.val_mask)
        self.test_data_list = create_with_mask(self.test_mask)

    def plot(self):
        if self.input_type == 'dense':
            graph = adj_to_edge_index(self.graphs)
        else:
            graph = self.graphs
        label = self.labels
        plot_graph(graph, label)

    def plot_expl(self, idx, expl, save_path=None, file_name=None, if_save=False, if_show=True):
        if if_save:
            if_show = False
        feature = self.features[idx]
        label = self.labels[idx]
        edge_ground_truth = self.edge_ground_truth[idx]
        node_ground_truth = self.node_ground_truth[idx]
        if self.input_type == 'dense':
            graph = adj_to_edge_index(self.graphs)[idx]
        else:
            graph = self.graphs[idx]
        plot_expl(graph, feature, label, edge_ground_truth, expl, save_path=save_path, file_name=file_name, if_save=if_save, if_show=if_show)
        # assert 0

    def save_expl(self, idx, expl, auc, save_path=None, file_name=None):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        idx = idx.tolist()
        expl = expl.tolist()
        try:
            auc = auc.tolist()
        except:
            auc = float(auc)
        feature = self.features[idx].tolist()
        label = self.labels[idx].tolist()
        edge_ground_truth = self.edge_ground_truth[idx].tolist()
        node_ground_truth = self.node_ground_truth[idx].tolist()
        if self.input_type == 'dense':
            graph = adj_to_edge_index(self.graphs)[idx].tolist()
        else:
            graph = self.graphs[idx].tolist()

        result = {
            'idx': idx,
            'graph': graph,
            'feature': feature,
            'label': label,
            'edge_ground_truth': edge_ground_truth,
            'node_ground_truth': node_ground_truth,
            'expl': expl,
            'auc': auc,
        }

        with open(os.path.join(save_path, file_name), 'w') as f:
            json.dump(result, f)

    def plot_expl_from_json(self, explainer_name, dataset_name, model_name, seed, idx, if_save=False, if_show=True):
        if if_save:
            if_show = False

        folder_path = os.path.join('data/results/visualization', explainer_name, dataset_name, model_name)

        for config in os.listdir(folder_path):
            config_path = os.path.join(folder_path, config)

            for i in os.listdir(config_path):
                if i.startswith(str(seed)+'_'):
                    config_path = os.path.join(config_path, i)
            for i in os.listdir(config_path):
                if i.startswith(str(idx)+'_'):
                    config_path = os.path.join(config_path, i)

            file_name = str(idx)+'.json'
            file_path = os.path.join(config_path, file_name)

            with open(file_path, 'r') as f:
                data = json.load(f)

            graph = data['graph']
            feature = data['feature']
            label = data['label'][0]
            edge_ground_truth = data['edge_ground_truth']
            node_ground_truth = data['node_ground_truth']
            expl = data['expl']

            plot_expl(graph, feature, label, edge_ground_truth, expl, save_path=config_path, file_name=str(idx)+'.png',
                      if_save=if_save, if_show=if_show)
        # assert 0

    def show_results(self, explainer_name, dataset_name, model_name):
        data = self.load_results(explainer_name, dataset_name, model_name)
        return data[1], data[2], data[3], data[4]

    def load_results(self, explainer_name, dataset_name, model_name):
        folder_path = os.path.join('/Users/jiaxingzhang/Desktop/Lab/XAI_GNN/XAIG_REG/data/results/visualization',
                                   explainer_name, dataset_name, model_name)
        confs = os.listdir(folder_path)
        best_auc = -1
        data = []
        for conf in confs:
            conf_path = os.path.join(folder_path, conf)
            seeds = os.listdir(conf_path)
            auc_s = []
            rdd_s = []
            for seed in seeds:
                if seed == '.DS_Store':
                    continue
                res = seed.split('_')
                idx = res[0]
                auc = float(res[1])
                rdd = float(res[2])
                auc_s.append(auc)
                rdd_s.append(rdd)
            auc = np.mean(auc_s)
            auc_std = np.std(auc_s)
            rdd = np.mean(rdd_s)
            rdd_std = np.std(rdd_s)
            if auc > best_auc:
                best_auc = auc
                data = [conf, auc, auc_std, rdd, rdd_std]
        return data
