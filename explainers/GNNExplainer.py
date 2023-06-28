from math import sqrt
import random
import numpy as np
import torch
import torch_geometric as ptgeom
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils.convert import from_scipy_sparse_matrix, to_scipy_sparse_matrix
from tqdm import tqdm

from explainers.BaseExplainer import BaseExplainer
from utils.dataset_util.data_utils import index_edge
from utils.wandb_logger import WandbLogger

"""
This is an adaption of the GNNExplainer of the PyTorch-Lightning library. 

The main similarity is the use of the methods _set_mask and _clear_mask to handle the mask. 
The main difference is the handling of different classification tasks. The original Geometric implementation only works for node 
classification. The implementation presented here also works for graph_classification datasets. 

link: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/gnn_explainer.html
"""


class GNNExplainer(BaseExplainer):
    """
    A class encaptulating the GNNexplainer (https://arxiv.org/abs/1903.03894).
    
    :param model_to_explain: graph classification model who's predictions we wish to explain.
    :param graphs: the collections of edge_indices representing the graphs
    :param features: the collcection of features for each node in the graphs.
    :param task: str "node" or "graph"
    :param model_type: str "cls" or "reg"
    :param epochs: amount of epochs to train our explainer
    :param lr: learning rate used in the training of the explainer
    :param reg_coefs: reguaization coefficients used in the loss. The first item in the tuple restricts the size of the explainations, the second rescticts the entropy matrix mask.
    
    :function __set_masks__: utility; sets learnable mask on the graphs.
    :function __clear_masks__: utility; rmoves the learnable mask.
    :function _loss: calculates the loss of the explainer
    :function explain: trains the explainer to return the subgraph which explains the classification of the model-to-be-explained.
    """

    def __init__(self, model_to_explain, graphs, features, task, model_type, loss_type, epochs=30, lr=0.003, reg_coefs=(0.03, 0.01)):
        super().__init__(model_to_explain, graphs, features, task, model_type, loss_type)
        self.epochs = epochs
        self.lr = lr
        self.reg_coefs = reg_coefs

        #  init logger
        self.config = {
            'epochs': self.epochs,
            'lr': self.lr,
            'reg_coefs': self.reg_coefs,
            'task': self.task,
            'type': self.model_type,
            'loss_name': self.loss_name,
        }

    def _set_masks(self, x, edge_index):
        """
        Inject the explanation maks into the message passing modules.
        :param x: features
        :param edge_index: graph representation
        """
        (N, F), E = x.size(), edge_index.size(1)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)

    def _clear_masks(self):
        """
        Cleans the injected edge mask from the message passing modules. Has to be called before any new sample can be explained.
        """
        self.edge_mask = None

    def prepare(self, args):
        """Nothing is done to prepare the GNNExplainer, this happens at every index"""
        return

    def explain(self, index):
        """
        Main method to construct the explanation for a given sample. This is done by training a mask such that the masked graph still gives
        the same prediction as the original graph using an optimization approach
        :param index: index of the node/graph that we wish to explain
        :return: explanation graph and edge weights
        """
        index = int(index)

        # Prepare model for new explanation run
        self.model_to_explain.eval()
        self._clear_masks()

        if self.task == 'node':
            # Similar to the original paper we only consider a subgraph for explaining
            feats = self.features
            graph = ptgeom.utils.k_hop_subgraph(index, 3, self.graphs)[1]
            # print(graph)
            with torch.no_grad():
                original_pred = self.model_to_explain(feats, graph)
                original_pred = original_pred[index]
                pred_label = original_pred.argmax(dim=-1).detach()
        else:
            feats = self.features[index].detach()
            graph = self.graphs[index].detach()
            # Remove self-loops
            graph = graph[:, (graph[0] != graph[1])]
            with torch.no_grad():
                original_pred = self.model_to_explain(feats, graph)
                # pred_label = original_pred.argmax(dim=-1).detach()
                pred_label = original_pred.detach()

        self._set_masks(feats, graph)
        optimizer = Adam([self.edge_mask], lr=self.lr)

        # Start training loop
        for e in range(0, self.epochs):
            optimizer.zero_grad()

            # Sample possible explanation
            if self.task == 'node':
                masked_pred = self.model_to_explain(feats, graph, edge_weights=torch.sigmoid(self.edge_mask))
                masked_pred = masked_pred[index]
                loss = self.loss(masked_pred.unsqueeze(0), pred_label.unsqueeze(0), self.edge_mask, self.reg_coefs)
            else:
                masked_pred = self.model_to_explain(feats, graph, edge_weights=torch.sigmoid(self.edge_mask))
                loss = self.loss(masked_pred, pred_label, self.edge_mask, self.reg_coefs)

            self.wandb_logger.log_step(self.loss_data, step=e)
            em_data = torch.sigmoid(self.edge_mask).detach().numpy().tolist()
            # em_data = {'weight'+str(i): em_data[i] for i in range(len(em_data))}
            em_data_dict = {}
            for i in range(len(em_data)):
                em_data_dict['weights/weight_'+str(i)] = em_data[i]
            self.wandb_logger.log_step(em_data_dict, step=e)
            self.total_step += 1
            loss.backward()
            optimizer.step()

        # Retrieve final explanation
        mask = torch.sigmoid(self.edge_mask)
        expl_graph_weights = torch.zeros(graph.size(1))
        for i in range(0, self.edge_mask.size(0)):  # Link explanation to original graph
            pair = graph.T[i]
            t = index_edge(graph, pair)
            expl_graph_weights[t] = mask[i]

        return graph, expl_graph_weights

    def serialize_configuration(self):
        return 'lr_' + str(self.lr) + '_epochs_' + str(self.epochs) + '_reg_coefs_' + str(self.reg_coefs[0]) + '_' + \
               str(self.reg_coefs[1]) + '_loss_' + self.loss_name
