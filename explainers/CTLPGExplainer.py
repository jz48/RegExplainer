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


class CTLPGExplainer(BaseExplainer):
    """
    A class encaptulating the PGExplainer (https://arxiv.org/abs/2011.04573).

    :param model_to_explain: graph classification model who's predictions we wish to explain.
    :param graphs: the collections of edge_indices representing the graphs.
    :param features: the collcection of features for each node in the graphs.
    :param task: str "node" or "graph".
    :param epochs: amount of epochs to train our explainer.
    :param lr: learning rate used in the training of the explainer.
    :param temp: the temperture parameters dictacting how we sample our random graphs.
    :param reg_coefs: reguaization coefficients used in the loss. The first item in the tuple restricts the size of the explainations, the second rescticts the entropy matrix mask.
    :params sample_bias: the bias we add when sampling random graphs.

    :function _create_explainer_input: utility;
    :function _sample_graph: utility; sample an explanatory subgraph.
    :function _loss: calculate the loss of the explainer during training.
    :function train: train the explainer
    :function explain: search for the subgraph which contributes most to the clasification decision of the model-to-be-explained.
    """

    def __init__(self, model_to_explain, graphs, features, task, model_type, loss_type, epochs=30, lr=0.0001,
                 temp=(5.0, 2.0),
                 reg_coefs=(0.05, 1.0), sample_bias=0):
        super().__init__(model_to_explain, graphs, features, task, model_type, loss_type)

        self.epochs = epochs
        self.lr = lr
        self.temp = temp
        self.reg_coefs = reg_coefs
        self.sample_bias = sample_bias
        self.dropoutlayer = nn.Dropout(0.1)
        self.lambda_ = 1.0
        self.ita = 1.0
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

    def _create_explainer_input(self, pair1, embeds1, node_id1, pair2, embeds2, node_id2):
        """
        Given the embeddign of the sample by the model that we wish to explain, this method construct the input to the mlp explainer model.
        Depending on if the task is to explain a graph or a sample, this is done by either concatenating two or three embeddings.
        :param pair: edge pair
        :param embeds: embedding of all nodes in the graph
        :param node_id: id of the node, not used for graph datasets
        :return: concatenated embedding
        """

        mask1 = []
        mask2 = []
        merge_edge_index = [[], []]
        mark_idx_1 = 0
        mark_idx_2 = 0
        edge_index1 = pair1
        edge_index2 = pair2

        def check_smaller_index(a1_, a2_, b1_, b2_):
            if a1_ < b1_:
                return True
            if a1_ > b1_:
                return False
            if a1_ == b1_:
                if a2_ < b2_:
                    return True
                else:
                    return False

        while True:
            a1 = edge_index1[0][mark_idx_1].item()
            b1 = edge_index2[0][mark_idx_2].item()
            a2 = edge_index1[1][mark_idx_1].item()
            b2 = edge_index2[1][mark_idx_2].item()
            if a1 == b1 and a2 == b2:
                src = a1
                tgt = a2
                merge_edge_index[0].append(src)
                merge_edge_index[1].append(tgt)
                mask1.append(1)
                mask2.append(1)
                mark_idx_1 += 1
                mark_idx_2 += 1
            elif check_smaller_index(a1, a2, b1, b2):
                src = a1
                tgt = a2
                merge_edge_index[0].append(src)
                merge_edge_index[1].append(tgt)
                mask1.append(1)
                mask2.append(0)
                mark_idx_1 += 1
            else:
                src = b1
                tgt = b2
                merge_edge_index[0].append(src)
                merge_edge_index[1].append(tgt)
                mask1.append(0)
                mask2.append(1)
                mark_idx_2 += 1
            if mark_idx_1 >= len(edge_index1[0]):
                while mark_idx_2 < len(edge_index2[0]):
                    src = edge_index2[0][mark_idx_2].item()
                    tgt = edge_index2[1][mark_idx_2].item()
                    merge_edge_index[0].append(src)
                    merge_edge_index[1].append(tgt)
                    mask1.append(0)
                    mask2.append(1)
                    mark_idx_2 += 1
                break
            if mark_idx_2 >= len(edge_index2[0]):
                while mark_idx_1 < len(edge_index1[0]):
                    src = edge_index1[0][mark_idx_1].item()
                    tgt = edge_index1[1][mark_idx_1].item()
                    merge_edge_index[0].append(src)
                    merge_edge_index[1].append(tgt)
                    mask1.append(1)
                    mask2.append(0)
                    mark_idx_1 += 1
                break

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
            if self.task == 'node':
                node_embed = embeds[node_id].repeat(rows.size(0), 1)
                input_expl = torch.cat([row_embeds, col_embeds, node_embed], 1)
            else:
                # Node id is not used in this case
                input_expl = torch.cat([row_embeds, col_embeds], 1)
            return input_expl

        self.mask1 = torch.tensor(mask1)
        self.mask2 = torch.tensor(mask2)
        self.merged_graph = torch.tensor(merge_edge_index)
        input_expl1 = build_input_expl(merge_edge_index, embeds1, node_id1, mask1)
        input_expl2 = build_input_expl(merge_edge_index, embeds2, node_id2, mask2)
        return input_expl1, input_expl2

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

        self.build_neighbors_sim_score(indices)
        self.train(indices=indices)

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

        # If we are explaining a graph, we can determine the embeddings before we run
        if self.task == 'node':
            embeds = self.model_to_explain.embedding(self.features, self.graphs).detach()

        # Start training loop
        for e in tqdm(range(0, self.epochs)):
            optimizer.zero_grad()
            loss = torch.FloatTensor([0]).detach()
            t = temp_schedule(e)

            for n in indices:
                n = int(n)

                near_index = self.sample_index(n, False)
                far_index = self.sample_index(n, True)
                near_prob = self.neighbor_matrix[n][near_index]
                far_prob = self.neighbor_matrix[n][far_index]
                if near_prob < far_prob:
                    near_index, far_index = far_index, near_index

                feats = self.features[n].detach()
                graph = self.graphs[n].detach()
                embeds = self.model_to_explain.embedding(feats, graph).detach()
                original_ebd = self.model_to_explain.build_graph_vector(feats, graph).detach()

                feats_near = self.features[near_index].detach()
                graph_near = self.graphs[near_index].detach()
                embeds_near = self.model_to_explain.embedding(feats_near, graph_near).detach()
                near_ebd = self.model_to_explain.build_graph_vector(feats_near, graph_near).detach()

                feats_far = self.features[far_index].detach()
                graph_far = self.graphs[far_index].detach()
                embeds_far = self.model_to_explain.embedding(feats_far, graph_far).detach()
                far_ebd = self.model_to_explain.build_graph_vector(feats_far, graph_far).detach()

                # Sample possible explanation
                # near
                input_expl1, input_expl2 = self._create_explainer_input(graph, embeds, n,
                                                                        graph_near, embeds_near, near_index)
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

                l_mask1 = torch.mul(mask1, self.lambda_)
                l_mask2 = torch.mul(mask2, self.lambda_)

                t2 = self.dropoutlayer(self.mask2 - l_mask2)
                mask_pred1 = torch.add(l_mask1, t2)
                mask_pred1 = torch.sigmoid(mask_pred1)

                t3 = self.dropoutlayer(self.mask1 - l_mask1)
                mask_pred2 = torch.add(l_mask2, t3)
                mask_pred2 = torch.sigmoid(mask_pred2)

                masked_pred1_near = self.model_to_explain(feats, self.merged_graph, edge_weights=mask_pred1)
                masked_ebd1 = self.model_to_explain.build_graph_vector(feats, self.merged_graph,
                                                                       edge_weights=mask_pred1)
                masked_pred2 = self.model_to_explain(feats_near, self.merged_graph, edge_weights=mask_pred2)
                original_pred = self.model_to_explain(feats, graph)
                original_pred2 = self.model_to_explain(feats_near, graph_near)

                # loss += self.Contrastive_loss(masked_ebd1, original_ebd)
                w = torch.tensor(self.w, requires_grad=False) # .to(device)

                loss += w * self.Contrastive_loss(masked_ebd1, near_ebd)
                id_loss = self.loss(masked_pred1_near, original_pred, mask_pred1, self.reg_coefs)
                loss += id_loss
                id_loss = self.loss(masked_pred2, original_pred2, mask_pred2, self.reg_coefs)
                loss += id_loss

                # far
                input_expl1, input_expl2 = self._create_explainer_input(graph, embeds, n,
                                                                        graph_far, embeds_far, far_index)
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

                l_mask1 = torch.mul(mask1, self.lambda_)
                l_mask2 = torch.mul(mask2, self.lambda_)

                t2 = self.dropoutlayer(self.mask2 - l_mask2)
                mask_pred1 = torch.add(l_mask1, t2)
                mask_pred1 = torch.sigmoid(mask_pred1)

                t3 = self.dropoutlayer(self.mask1 - l_mask1)
                mask_pred2 = torch.add(l_mask2, t3)
                mask_pred2 = torch.sigmoid(mask_pred2)

                masked_pred1_far = self.model_to_explain(feats, self.merged_graph, edge_weights=mask_pred1)
                masked_ebd2 = self.model_to_explain.build_graph_vector(feats, self.merged_graph,
                                                                       edge_weights=mask_pred1)
                masked_pred2 = self.model_to_explain(feats_far, self.merged_graph, edge_weights=mask_pred2)
                original_pred = self.model_to_explain(feats, graph)
                original_pred2 = self.model_to_explain(feats_far, graph_far)

                # loss = loss - self.Contrastive_loss(original_ebd, masked_ebd2)
                loss = loss - w * self.Contrastive_loss(masked_ebd2, far_ebd)
                # loss = loss + self.cross_inner_product_loss(masked_pred1_near, masked_pred1_far)

                # loss += self.Contrastive_loss(masked_ebd1, original_ebd)
                # id_loss = self.loss(masked_pred1_far, original_pred, mask_pred1, self.reg_coefs)
                # loss += id_loss
                # id_loss = self.loss(masked_pred2, original_pred2, mask_pred2, self.reg_coefs)
                # loss += id_loss

            loss.backward()
            optimizer.step()

    def explain(self, index):
        """
        Given the index of a node/graph this method returns its explanation. This only gives sensible results if the prepare method has
        already been called.
        :param index: index of the node/graph that we wish to explain
        :return: explanaiton graph and edge weights
        """
        index = int(index)
        while True:
            index2 = int(random.randint(0, len(self.graphs) - 1))
            if index2 != index:
                break
        if self.task == 'node':
            # Similar to the original paper we only consider a subgraph for explaining
            graph = ptgeom.utils.k_hop_subgraph(index, 3, self.graphs)[1]
            embeds = self.model_to_explain.embedding(self.features, self.graphs).detach()
        else:
            feats = self.features[index].clone().detach()
            graph = self.graphs[index].clone().detach()
            embeds = self.model_to_explain.embedding(feats, graph).detach()

            feats2 = self.features[index2].detach()
            graph2 = self.graphs[index2].detach()
            embeds2 = self.model_to_explain.embedding(feats2, graph2).detach()

        # Use explainer mlp to get an explanation
        input_expl, _ = self._create_explainer_input(graph, embeds, index, graph2, embeds2, index2)
        input_expl = input_expl.unsqueeze(dim=0)
        sampling_weights = self.explainer_model(input_expl)
        mask = self._sample_graph(sampling_weights, training=False).squeeze()
        final_mask = []
        for i in range(len(self.mask1)):
            # print(self.mask1[i])
            if self.mask1[i] == 1:
                final_mask.append(mask[i])
        expl_graph_weights = torch.zeros(graph.size(1))  # Combine with original graph
        for i in range(0, len(final_mask)):
            pair = graph.T[i]
            t = index_edge(graph, pair)
            expl_graph_weights[t] = final_mask[i]

        return graph, expl_graph_weights

    def serialize_configuration(self):
        return 'lr_' + str(self.lr) + '_epochs_' + str(self.epochs) + '_reg_coefs_' + str(self.reg_coefs[0]) + '_' + \
               str(self.reg_coefs[1]) + '_sample_bias_' + str(self.sample_bias) + '_temp_' + str(self.temp[0]) + \
               str(self.temp[1]) + '_loss_' + self.loss_name + '_lambda_' + str(self.lambda_) + '_ita_' + \
               str(self.ita) + '_w_' + str(self.w) + '_alpha_' + str(self.w) + \
               str(self.ita) + '_beta_' + str(self.beta)
