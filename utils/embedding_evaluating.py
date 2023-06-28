import json
import math
import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data, DataLoader
from utils.dataset_util.dataset_loaders import DatasetLoader
from utils.dataset_util.data_utils import to_torch_graph
from gnns.model_selector import ModelSelector
from explainers.explainer_selector import ExplainerSelector
from utils.explaining import MainExplainer


class MainEvaluator(MainExplainer):
    def __init__(self, dataset_name, model_name, explainer_name, wandb_log=False, loss_type='ib',
                 sample_bias=0.0, thres_snip=5, thres_min=-1
                 , temps=None, seeds=None, save_explaining_results=True, force_run=False):
        super().__init__(dataset_name, model_name, explainer_name, wandb_log=False, loss_type='ib',
                         sample_bias=0.0, thres_snip=5, thres_min=-1
                         , temps=None, seeds=None, save_explaining_results=True, force_run=False)
        self.sub_graphs = self.transform_ground_truth_to_sub_graph()

    def transform_ground_truth_to_sub_graph(self):
        sub_graphs = []
        for i in range(len(self.graphs)):
            graph = self.graphs[i]
            ground_truth = self.ground_truth[i]
            sub_graph = [[], []]
            a = 0
            for j in range(len(ground_truth)):
                if ground_truth[j] == 1:
                    sub_graph[0].append(graph[0][j])
                    sub_graph[1].append(graph[1][j])
            sub_graphs.append(np.array(sub_graph))
        return sub_graphs

    def explain(self):
        a = 0
        self.explainer_manager.explainer.features = torch.tensor(self.features)
        self.explainer_manager.explainer.labels = torch.tensor(self.labels)
        self.explainer_manager.explainer.graphs = to_torch_graph(self.graphs, self.explainer_manager.task)

        # Get ground_truth for every node
        indices = np.argwhere(self.dataset_loader.val_mask).squeeze()

        # Perform the evaluation 10 times
        auc_scores = []

        for turn_, s in enumerate(self.seeds):
            print(f"Run {turn_} with seed {s}")
            # Set all seeds needed for reproducibility
            torch.manual_seed(s)
            torch.cuda.manual_seed(s)
            np.random.seed(s)

            self.explainer_manager.explainer.prepare(indices)

            explanations = []
            for idx in tqdm(indices):
                graph, expl = self.explainer_manager.explainer.explain(idx)
                explanations.append((idx, graph, expl))

            auc_score = self.evaluation_auc(explanations)
            print("score:", auc_score)

            for idx, graph_, expl_ in explanations:
                graph = graph_.detach().numpy()
                expl = expl_.detach().numpy()
                i_auc = self.evaluation_auc_single_graph(idx, graph, expl)
                folder_path = os.path.join('./data/results/visualization/',
                                           self.explainer_name, self.dataset_name, self.model_name,
                                           str(s) + '_' + str(auc_score), str(idx) + '_' + str(i_auc))
                file_name = str(idx) + '.json'
                self.dataset_loader.save_expl(idx, expl, i_auc, save_path=folder_path,
                                              file_name=file_name)
            auc_scores.append(auc_score)

        auc = np.mean(auc_scores)
        auc_std = np.std(auc_scores)

        return auc, auc_std

    def embedding_evaluate(self):
        self.explainer_manager.explainer.features = torch.tensor(self.features)
        self.explainer_manager.explainer.labels = torch.tensor(self.labels)
        self.explainer_manager.explainer.graphs = to_torch_graph(self.graphs, self.explainer_manager.task)
        sub_graphs = [torch.tensor(i) for i in self.sub_graphs]  # torch.tensor(self.sub_graphs)
        # Get ground_truth for every node
        indices = np.argwhere(self.dataset_loader.val_mask).squeeze()

        # Perform the evaluation 10 times
        auc_scores = []

        results = {}
        for turn_, s in enumerate(self.seeds):
            print(f"Run {turn_} with seed {s}")
            # Set all seeds needed for reproducibility
            torch.manual_seed(s)
            torch.cuda.manual_seed(s)
            np.random.seed(s)

            self.explainer_manager.explainer.prepare(indices)

            embeddings = []
            for idx in tqdm(indices):
                index = int(idx)
                feats = self.explainer_manager.explainer.features[index]
                graph = self.explainer_manager.explainer.graphs[index]
                sub_graph = sub_graphs[index]

                graph_ebd = self.explainer_manager.explainer.model_to_explain.build_graph_vector(feats,
                                                                                                 graph).detach().numpy().tolist()
                sub_graph_ebd = self.explainer_manager.explainer.model_to_explain.build_graph_vector(feats,
                                                                                                     sub_graph).detach().numpy().tolist()

                embeddings.append([int(idx), graph_ebd, sub_graph_ebd])

            results[int(s)] = embeddings

        file_name = self.explainer_name + '_' + self.dataset_name + '_' + self.model_name + '.json'
        file_path = os.path.join('./data/results/embeddings_evaluation/', file_name)
        with open(file_path, 'w') as f:
            json.dump(results, f)
        return 0, 0

    def evaluation_auc(self, explanations):
        """Determines based on the task which auc evaluation method should be called to determine the AUC score

        :param task: str either "node" or "graph".
        :param explanations: predicted labels.
        :param ground_truth: ground truth labels.
        :param indices: Which indices to evaluate. We ignore all others.
        :returns: area under curve score.
        """
        if self.explainer_manager.explainer.task == 'graph':
            return self.evaluation_auc_graph(explanations)
        elif self.explainer_manager.explainer.task == 'node':
            return self.evaluation_auc_node(explanations)

    def evaluation_auc_graph(self, explanations):
        """Evaluate the auc score given explaination and ground truth labels.

        :param explanations: predicted labels.
        :param ground_truth: ground truth labels.
        :param indices: Which indices to evaluate. We ignore all others.
        :returns: area under curve score.
        """
        ys = []
        predictions = []

        for i in explanations:
            idx = i[0]
            graph = i[1].detach().numpy()
            expl = i[2].detach().numpy()
            ground_truth = self.dataset_loader.edge_ground_truth[idx]

            for edge_idx in range(0, expl.shape[0]):  # Consider every edge in the ground truth
                edge_ = graph.T[edge_idx]
                if edge_[0] == edge_[1]:  # We dont consider self loops for our evaluation (Needed for Ba2Motif)
                    continue

                # Retrieve predictions and ground truth
                predictions.append(expl[edge_idx])
                ys.append(ground_truth[edge_idx])
            a = 0

        score = roc_auc_score(ys, predictions)
        return score

    def evaluation_auc_single_graph(self, idx, graph, expl):
        """Evaluate the auc score given explaination and ground truth labels.

        :param graph: graph
        :param expl: predicted labels.
        :param idx: ground truth labels.
        :returns: area under curve score.
        """
        ys = []
        predictions = []

        ground_truth = self.dataset_loader.edge_ground_truth[idx]

        for edge_idx in range(0, expl.shape[0]):  # Consider every edge in the ground truth
            edge_ = graph.T[edge_idx]
            if edge_[0] == edge_[1]:  # We dont consider self loops for our evaluation (Needed for Ba2Motif)
                continue
            # Retrieve predictions and ground truth
            predictions.append(expl[edge_idx])
            ys.append(ground_truth[edge_idx])

        score = roc_auc_score(ys, predictions)
        return score

    def evaluation_auc_node(self, explanations):
        """Evaluate the auc score given explaination and ground truth labels.

        :param explanations: predicted labels.
        :param ground_truth: ground truth labels.
        :param indices: Which indices to evaluate. We ignore all others.
        :returns: area under curve score.
        """
        ground_truth = []
        predictions = []
        for expl in explanations:  # Loop over the explanations for each node

            ground_truth_node = []
            prediction_node = []

            for i in range(0, expl[0].size(1)):  # Loop over all edges in the explanation sub-graph
                prediction_node.append(expl[1][i].item())

                # Graphs are defined bidirectional, so we need to retrieve both edges
                pair = expl[0].T[i].numpy()
                idx_edge = np.where((explanation_labels[0].T == pair).all(axis=1))[0]
                idx_edge_rev = np.where((explanation_labels[0].T == [pair[1], pair[0]]).all(axis=1))[0]

                # If any of the edges is in the ground truth set, the edge should be in the explanation
                gt = explanation_labels[1][idx_edge] + explanation_labels[1][idx_edge_rev]
                if gt == 0:
                    ground_truth_node.append(0)
                else:
                    ground_truth_node.append(1)

            ground_truth.extend(ground_truth_node)
            predictions.extend(prediction_node)

        score = roc_auc_score(ground_truth, predictions)
        return score


def main():
    pass


if __name__ == '__main__':
    main()
