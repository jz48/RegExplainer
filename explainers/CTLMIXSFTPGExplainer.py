import torch
import torch_geometric as ptgeom
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_scipy_sparse_matrix, to_scipy_sparse_matrix
from torch_sparse import SparseTensor
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import random
import numpy as np
import time
from explainers.BaseExplainer import BaseExplainer
from utils.dataset_util.data_utils import index_edge


class CTLMixUpSFTPGExplainer(BaseExplainer):
    def __init__(self, model_to_explain, graphs, features, task, model_type, loss_type, epochs=30, lr=0.003, temp=(5.0, 2.0),
                 reg_coefs=(0.05, 1.0), sample_bias=0):
        super().__init__(model_to_explain, graphs, features, task, model_type, loss_type)

        self.epochs = epochs
        self.lr = lr
        self.temp = temp
        self.reg_coefs = reg_coefs
        self.sample_bias = sample_bias
        self.dropoutlayer = nn.Dropout(0.1)
        self.lambda_ = 1.0
        self.ita = 0.03
        self.w = 1.0  # weight for contrastive loss
        self.config = {
            'epochs': self.epochs,
            'lr': self.lr,
            'temp': self.temp,
            'reg_coefs': self.reg_coefs,
            'sample_bias': self.sample_bias,
            'task': self.task,
            'type': self.model_type,
            'loss_name': self.loss_name,
            'lambda': self.lambda_,
            'ita': self.ita,
            'w': self.w
        }

        if self.task == "graph":
            self.expl_embedding = self.model_to_explain.embedding_size * 2
        else:
            self.expl_embedding = self.model_to_explain.embedding_size * 3

    # @func_timer
    def _create_explainer_input(self, pair1, embeds1, node_id1, pair2, embeds2, node_id2):
        """
        Given the embedding of the sample by the model that we wish to explain, this method construct the input to the mlp explainer model.
        Depending on if the task is to explain a graph or a sample, this is done by either concatenating two or three embeddings.
        :param pair: edge pair
        :param embeds: embedding of all nodes in the graph
        :param node_id: id of the node, not used for graph datasets
        :return: concatenated embedding
        """

        mask1 = []
        mask2 = []
        edge_index1 = pair1
        edge_index2 = pair2

        # @func_timer
        def adj2edge_list(adj):
            # return from_scipy_sparse_matrix(adj)
            adj = SparseTensor.from_dense(torch.tensor(adj))
            row, col, edge_attr = adj.t().coo()
            edge_index = torch.stack([row, col], dim=0)
            return edge_index
            edge_list = [[], []]
            for row in range(adj.shape[0]):
                for col in range(adj.shape[1]):
                    if adj[row][col] == 1:
                        edge_list[0].append(row)
                        edge_list[1].append(col)
            return edge_list

        # @func_timer
        def edge_list2adj(edge_list, size_adj):
            # adj = to_scipy_sparse_matrix(edge_list)
            adj = np.zeros((size_adj, size_adj), dtype=float)
            for i in range(len(edge_list[0])):
                adj[edge_list[0][i]][edge_list[1][i]] = 1.0
            return adj

        # t0 = time.time()
        self.index1 = self.index1
        self.index2 = self.index2 + embeds1.size(0)
        adj1 = edge_list2adj(edge_index1, embeds1.size(0))
        adj2 = edge_list2adj(edge_index2, embeds2.size(0))

        # t1 = time.time()
        link_adj1 = np.zeros((adj1.shape[0], adj2.shape[0]))
        # t2 = time.time()
        # print('time for create link_adj1: ', t2 - t1)

        yita = self.ita  # / (adj1.shape[0] * adj2.shape[0])
        # yita = 0.03
        for i in range(link_adj1.shape[0]):
            for j in range(link_adj1.shape[1]):
                if random.random() < yita:
                    link_adj1[i][j] = 1.0
        # t3 = time.time()
        # print('time for build link_adj1: ', t3 - t2)

        link_adj2 = link_adj1.T
        # t4 = time.time()
        # print('time for transpose link_adj1: ', t4 - t3)

        a = np.concatenate((adj1, link_adj1), axis=1)
        b = np.concatenate((link_adj2, adj2), axis=1)

        merged_adj = np.concatenate((a, b), axis=0)
        merge_edge_index = adj2edge_list(merged_adj)
        merged_feats = torch.cat((self.feats1, self.feats2), 0)
        merged_embeds = torch.cat((embeds1, embeds2), 0)
        # t5 = time.time()
        for i in range(len(merge_edge_index[0])):
            if merge_edge_index[0][i] < embeds1.size(0) and merge_edge_index[1][i] < embeds1.size(0):
                mask1.append(1)
            else:
                mask1.append(0)
            if merge_edge_index[0][i] >= embeds1.size(0) or merge_edge_index[1][i] >= embeds1.size(0):
                mask2.append(1)
            else:
                mask2.append(0)
        # t6 = time.time()
        # print('time for create masks: ', t6 - t5)

        # @func_timer
        def build_input_expl(edge_index, embeds, node_id, mask):
            rows = edge_index[0]
            cols = edge_index[1]

            row_embeds = embeds[rows]
            col_embeds = embeds[cols]

            zeros = torch.zeros(len(embeds[0]))
            for i in range(len(row_embeds)):
                if mask[i] == 0:
                    row_embeds[i] = zeros.clone()
            for i in range(len(col_embeds)):
                if mask[i] == 0:
                    col_embeds[i] = zeros.clone()

            input_expl = torch.cat([row_embeds, col_embeds], 1)
            return input_expl

        self.mask1 = torch.tensor(mask1)
        self.mask2 = torch.tensor(mask2)
        self.merged_feats = merged_feats
        self.merged_graph = torch.tensor(merge_edge_index)
        input_expl1 = build_input_expl(merge_edge_index, merged_embeds, self.index1, mask1)
        input_expl2 = build_input_expl(merge_edge_index, merged_embeds, self.index2, mask2)
        return input_expl1, input_expl2

    # @func_timer
    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        """
        Implementation of the reparamerization trick to obtain a sample graph while maintaining the posibility to backprop.
        :param sampling_weights: Weights provided by the mlp
        :param temperature: annealing temperature to make the procedure more deterministic
        :param bias: Bias on the weights to make samplign less deterministic
        :param training: If set to false, the samplign will be entirely deterministic
        :return: sample graph
        """
        if training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(sampling_weights.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph = torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph

    def prepare(self, indices=None):
        """
        Before we can use the explainer we first need to train it. This is done here.
        :param indices: Indices over which we wish to train.
        """
        # Creation of the explainer_model is done here to make sure that the seed is set
        self.explainer_model = nn.Sequential(
            nn.Linear(self.expl_embedding, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        if indices is None:  # Consider all indices
            indices = range(0, self.graphs.size(0))

        self.neighbor_pairs = {}
        self.neighbor_matrix = [[] for _ in range(len(self.graphs))]
        for index in range(len(self.graphs)):
            feats = self.features[index].detach()
            graph = self.graphs[index].detach()
            with torch.no_grad():
                # original_pred1 = self.model_to_explain(feats, graph)
                # pred_label1 = original_pred1.detach()
                original_ebd = self.model_to_explain.build_graph_vector(feats, graph).detach().numpy()
            self.neighbor_pairs[index] = [original_ebd, -1]
        for _, key in enumerate(self.neighbor_pairs):
            for _, key2 in enumerate(self.neighbor_pairs):
                if key == key2:
                    self.neighbor_matrix[key].append(1.0)
                    continue
                if key2 < key:
                    self.neighbor_matrix[key].append(self.neighbor_matrix[key2][key])
                    continue
                cos_sim = cosine_similarity(self.neighbor_pairs[key][0], self.neighbor_pairs[key2][0])
                self.neighbor_matrix[key].append(cos_sim.tolist()[0][0])

        self.train(indices=indices)

    def sample_index(self, index_, if_reverse):
        weight_list = self.neighbor_matrix[index_]
        if if_reverse:
            weight_list = [1.0 - i_ for i_ in weight_list]
        indices = [i_ for i_ in range(len(weight_list))]
        while True:
            target = random.choices(indices, weights=weight_list, k=1)[0]
            if target != index_:
                break
        return target

    def train(self, indices=None):
        """
        Main method to train the model
        :param indices: Indices that we want to use for training.
        :return:
        """
        # Make sure the explainer model can be trained
        self.explainer_model.train()

        # Create optimizer and temperature schedule
        optimizer = Adam(self.explainer_model.parameters(), lr=self.lr)
        temp_schedule = lambda e: self.temp[0] * ((self.temp[1] / self.temp[0]) ** (e / self.epochs))

        # Start training loop
        for e in tqdm(range(0, self.epochs)):
            optimizer.zero_grad()
            loss = torch.FloatTensor([0]).detach()
            t = temp_schedule(e)

            for n in indices:
                n = int(n)
                self.index1 = n

                near_index = self.sample_index(n, False)
                far_index = self.sample_index(n, True)
                near_prob = self.neighbor_matrix[n][near_index]
                far_prob = self.neighbor_matrix[n][far_index]
                if near_prob < far_prob:
                    near_index, far_index = far_index, near_index

                feats1 = self.features[n].detach()
                self.feats1 = feats1
                graph1 = self.graphs[n].detach()
                graph1 = graph1[:, (graph1[0] != graph1[1])]
                embeds1 = self.model_to_explain.embedding(feats1, graph1).detach()
                original_ebd = self.model_to_explain.build_graph_vector(feats1, graph1).detach()

                # near
                self.index2 = near_index
                feats2 = self.features[near_index].detach()
                self.feats2 = feats2
                graph2 = self.graphs[near_index].detach()
                graph2 = graph2[:, (graph2[0] != graph2[1])]
                embeds2 = self.model_to_explain.embedding(feats2, graph2).detach()
                near_ebd = self.model_to_explain.build_graph_vector(feats2, graph2).detach()

                # Sample possible explanation
                input_expl1, input_expl2 = self._create_explainer_input(graph1, embeds1, n, graph2, embeds2, near_index)
                # assert 0
                input_expl1 = input_expl1.unsqueeze(0)
                input_expl2 = input_expl2.unsqueeze(0)
                sampling_weights1 = self.explainer_model(input_expl1)
                # sampling_weights1 = torch.mul(sampling_weights1, self.mask1)
                sampling_weights2 = self.explainer_model(input_expl2)
                # sampling_weights2 = torch.mul(sampling_weights2, self.mask2)
                mask1 = self._sample_graph(sampling_weights1, t, bias=self.sample_bias).squeeze()
                mask2 = self._sample_graph(sampling_weights2, t, bias=self.sample_bias).squeeze()

                lam = torch.tensor(self.lambda_, requires_grad=False)
                mask1 = torch.mul(mask1, self.mask1)
                mask2 = torch.mul(mask2, self.mask2)

                t2 = self.dropoutlayer(self.mask2 - lam * mask2)
                mask_pred1 = torch.add(lam * mask1, t2)
                mask_pred1 = torch.sigmoid(mask_pred1)
                t3 = self.dropoutlayer(self.mask1 - lam * mask1)
                mask_pred2 = torch.add(lam * mask2, t3)
                mask_pred2 = torch.sigmoid(mask_pred2)
                masked_pred1_near = self.model_to_explain(self.merged_feats, self.merged_graph, edge_weights=mask_pred1)
                masked_ebd1 = self.model_to_explain.build_graph_vector(self.merged_feats, self.merged_graph,
                                                                       edge_weights=mask_pred1)
                masked_pred2 = self.model_to_explain(self.merged_feats, self.merged_graph, edge_weights=mask_pred2)
                original_pred1 = self.model_to_explain(feats1, graph1)
                original_pred2 = self.model_to_explain(feats2, graph2)

                w = torch.tensor(self.w, requires_grad=False)

                # loss += self.Contrastive_loss(masked_ebd1, original_ebd)
                loss += w * self.Contrastive_loss(masked_ebd1, near_ebd)
                id_loss = self._loss(masked_pred1_near, original_pred1, mask_pred1, self.reg_coefs)
                loss += id_loss
                id_loss = self._loss(masked_pred2, original_pred2, mask_pred2, self.reg_coefs)
                loss += id_loss

                # far
                self.index2 = far_index
                feats2 = self.features[far_index].detach()
                self.feats2 = feats2
                graph2 = self.graphs[far_index].detach()
                graph2 = graph2[:, (graph2[0] != graph2[1])]
                embeds2 = self.model_to_explain.embedding(feats2, graph2).detach()
                far_ebd = self.model_to_explain.build_graph_vector(feats2, graph2).detach()

                # Sample possible explanation
                input_expl1, input_expl2 = self._create_explainer_input(graph1, embeds1, n, graph2, embeds2, near_index)
                # assert 0
                input_expl1 = input_expl1.unsqueeze(0)
                input_expl2 = input_expl2.unsqueeze(0)
                sampling_weights1 = self.explainer_model(input_expl1)
                # sampling_weights1 = torch.mul(sampling_weights1, self.mask1)
                sampling_weights2 = self.explainer_model(input_expl2)
                # sampling_weights2 = torch.mul(sampling_weights2, self.mask2)
                mask1 = self._sample_graph(sampling_weights1, t, bias=self.sample_bias).squeeze()
                mask2 = self._sample_graph(sampling_weights2, t, bias=self.sample_bias).squeeze()
                mask1 = torch.mul(mask1, self.mask1)
                mask2 = torch.mul(mask2, self.mask2)

                t2 = self.dropoutlayer(self.mask2 - lam * mask2)
                mask_pred1 = torch.add(lam * mask1, t2)
                mask_pred1 = torch.sigmoid(mask_pred1)
                t3 = self.dropoutlayer(self.mask1 - lam * mask1)
                mask_pred2 = torch.add(lam * mask2, t3)
                mask_pred2 = torch.sigmoid(mask_pred2)
                masked_pred1_far = self.model_to_explain(self.merged_feats, self.merged_graph, edge_weights=mask_pred1)
                masked_ebd1 = self.model_to_explain.build_graph_vector(self.merged_feats, self.merged_graph,
                                                                       edge_weights=mask_pred1)
                masked_pred2 = self.model_to_explain(self.merged_feats, self.merged_graph, edge_weights=mask_pred2)
                original_pred1 = self.model_to_explain(feats1, graph1)
                original_pred2 = self.model_to_explain(feats2, graph2)

                loss -= w * self.Contrastive_loss(masked_ebd1, far_ebd)
                # loss += self.cross_inner_product_loss(masked_pred1_near, masked_pred1_far)

                # id_loss = self._loss(masked_pred1_far, original_pred1, mask_pred1, self.reg_coefs)
                # loss += id_loss
                # id_loss = self._loss(masked_pred2, original_pred2, mask_pred2, self.reg_coefs)
                # loss += id_loss
            loss.backward()
            optimizer.step()

    # @func_timer
    def explain(self, index1):
        """
        Given the index of a node/graph this method returns its explanation. This only gives sensible results if the prepare method has
        already been called.
        :param index: index of the node/graph that we wish to explain
        :return: explanaiton graph and edge weights
        """
        index1 = int(index1)
        index2 = self.sample_index(index1, False)

        self.index1 = index1
        self.index2 = int(index2)

        if self.task == 'node':
            # Similar to the original paper we only consider a subgraph for explaining
            graph1 = ptgeom.utils.k_hop_subgraph(index1, 3, self.graphs)[1]
            embeds1 = self.model_to_explain.embedding(self.features, self.graphs).detach()

            graph2 = ptgeom.utils.k_hop_subgraph(index2, 3, self.graphs)[1]
            embeds2 = self.model_to_explain.embedding(self.features, self.graphs).detach()

        else:
            feats1 = self.features[index1].clone().detach()
            self.feats1 = feats1
            graph1 = self.graphs[index1].clone().detach()
            graph1 = graph1[:, (graph1[0] != graph1[1])]
            embeds1 = self.model_to_explain.embedding(feats1, graph1).detach()

            feats2 = self.features[index2].detach()
            self.feats2 = feats2
            graph2 = self.graphs[index2].detach()
            graph2 = graph2[:, (graph2[0] != graph2[1])]
            embeds2 = self.model_to_explain.embedding(feats2, graph2).detach()

        # Use explainer mlp to get an explanation
        input_expl, _ = self._create_explainer_input(graph1, embeds1, index1, graph2, embeds2, index2)
        input_expl = input_expl.unsqueeze(dim=0)
        sampling_weights = self.explainer_model(input_expl)
        mask = self._sample_graph(sampling_weights, training=False).squeeze()
        final_mask = []
        for i in range(len(self.mask1)):
            # print(self.mask1[i])
            if self.mask1[i] == 1:
                final_mask.append(mask[i])
        expl_graph_weights = torch.zeros(graph1.size(1))  # Combine with original graph
        for i in range(0, len(final_mask)):
            pair = graph1.T[i]
            t = index_edge(graph1, pair)
            expl_graph_weights[t] = final_mask[i]

        return graph1, expl_graph_weights

    def serialize_configuration(self):
        return 'lr_' + str(self.lr) + '_epochs_' + str(self.epochs) + '_reg_coefs_' + str(self.reg_coefs[0]) + '_' + \
               str(self.reg_coefs[1]) + '_sample_bias_' + str(self.sample_bias) + '_temp_' + str(self.temp[0]) + \
               str(self.temp[1]) + '_loss_' + self.loss_name + '_lambda_' + str(self.lambda_) + '_ita_' + \
               str(self.ita) + '_w_' + str(self.w) + str(self.ita) + '_alpha_' + str(self.w) + \
               str(self.ita) + '_beta_' + str(self.beta)
