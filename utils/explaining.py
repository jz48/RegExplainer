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
from utils.wandb_logger import WandbLogger


class MainExplainer:
    def __init__(self, dataset_name, model_name, explainer_name, wandb_log=False, loss_type='ib',
                 sample_bias=0.0, thres_snip=5, thres_min=-1
                 , temps=None, seeds=None, save_explaining_results=True, force_run=False):

        if seeds is None:
            self.seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        if temps is None:
            self.temps = [5.0, 1.0]

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.explainer_name = explainer_name
        self.logging = wandb_log
        self.loss_type = loss_type

        self.seed = seeds
        self.force_run = force_run
        self.save_explaining_results = save_explaining_results

        if model_name == 'RegGCN':
            self.input_type = 'dense'
        else:
            self.input_type = 'sparse'

        if dataset_name in ['bareg1', 'bareg2', 'crippen', 'bareg3', 'triangles', 'triangles_small']:
            self.type = 'reg'
        else:
            self.type = 'cls'

        self.dataset_loader = DatasetLoader(self.dataset_name, self.input_type, self.type)
        self.dataset_loader.load_dataset()
        self.dataset_loader.create_data_list()

        self.graphs, self.features = self.dataset_loader.graphs, self.dataset_loader.features
        self.labels, self.test_mask = self.dataset_loader.labels, self.dataset_loader.test_mask
        self.ground_truth = self.dataset_loader.edge_ground_truth

        self.model_manager = ModelSelector(model_name, dataset_name, load_pretrain=True)
        self.model_manager.model.type = self.type
        self.explainer_manager = ExplainerSelector(explainer_name, model_name, dataset_name, self.model_manager.model,
                                                   self.loss_type, self.graphs, self.features)
        project_name = explainer_name + '_' + dataset_name + '_' + model_name
        self.explainer_manager.explainer.wandb_logger = WandbLogger(project_name, self.explainer_manager.explainer.config, wandb_log)
        self.explainer_manager.explainer.total_step = 0

    def explain(self):
        if self.dataset_name in ['crippen']:
            self.explainer_manager.explainer.features = [torch.tensor(i) for i in self.features]
        else:
            self.explainer_manager.explainer.features = torch.tensor(self.features)

        self.explainer_manager.explainer.labels = torch.tensor(self.labels)
        self.explainer_manager.explainer.graphs = to_torch_graph(self.graphs, self.explainer_manager.task)

        # Get ground_truth for every node
        indices = np.argwhere(self.dataset_loader.val_mask).squeeze()
        if self.logging:
            self.seeds = [0, 1, 2]
            indices = indices[:10]
        # Perform the evaluation 10 times
        auc_scores = []
        rdd_scores = []

        folder_path = os.path.join(self.dataset_loader.data_root_path, 'results/visualization/',
                                   self.explainer_name, self.dataset_name, self.model_name,
                                   self.explainer_manager.explainer.serialize_configuration())
        if os.path.exists(folder_path) and not self.force_run:
            print('already finished!')
            return self.dataset_loader.show_results(self.explainer_name, self.dataset_name, self.model_name)

        save_score_path = os.path.join(self.dataset_loader.data_root_path, 'results/scores/',
                                       self.explainer_name, self.dataset_name, self.model_name,
                                       self.explainer_manager.explainer.serialize_configuration())

        if not os.path.exists(save_score_path):
            os.makedirs(save_score_path)

        scores = []
        for turn_, s in enumerate(self.seeds):
            print(f"Run {turn_} with seed {s}")
            # Set all seeds needed for reproducibility
            torch.manual_seed(s)
            torch.cuda.manual_seed(s)
            np.random.seed(s)

            self.explainer_manager.explainer.prepare(indices)

            explanations = []
            for idx in tqdm(indices, disable=(not self.logging)):
                self.explainer_manager.explainer.wandb_logger.start_log(s, idx)
                graph, expl = self.explainer_manager.explainer.explain(idx)
                explanations.append((idx, graph, expl))
                self.explainer_manager.explainer.wandb_logger.close()

            y_true, y_pred = self.prepare_evaluate(explanations)
            auc_score = self.evaluate_auc(y_true, y_pred)
            rdd_score = self.evaluate_rdd(y_true, y_pred)
            print("auc score:", auc_score, "rdd score: ", rdd_score)

            scores.append([s, auc_score, rdd_score])

            if self.save_explaining_results:
                for idx, graph_, expl_ in explanations:
                    y_true, y_pred = self.prepare_evaluate([(idx, graph_, expl_)])
                    try:
                        i_auc = self.evaluate_auc(y_true, y_pred)
                    except:
                        i_auc = 1.1
                    i_rdd = self.evaluate_rdd(y_true, y_pred)
                    folder_path = os.path.join(self.dataset_loader.data_root_path, 'results/visualization/',
                                               self.explainer_name, self.dataset_name, self.model_name,
                                               self.explainer_manager.explainer.serialize_configuration(),
                                               str(s) + '_' + str(auc_score) + '_' + str(rdd_score),
                                               str(idx) + '_' + str(i_auc) + '_' + str(i_rdd))
                    file_name = str(idx) + '.json'
                    self.dataset_loader.save_expl(idx, expl_.detach().numpy(), i_auc, save_path=folder_path,
                                                  file_name=file_name)

            auc_scores.append(auc_score)
            rdd_scores.append(rdd_score)

        with open(os.path.join(save_score_path, 'scores.json'), 'w') as f:
            f.write(json.dumps(scores))

        auc = np.mean(auc_scores)
        auc_std = np.std(auc_scores)
        rdd = np.mean(rdd_scores)
        rdd_std = np.std(rdd_scores)
        return auc, auc_std, rdd, rdd_std

    def prepare_evaluate(self, explanations):
        y_true = []
        y_pred = []
        for idx, graph_, expl_ in explanations:
            graph = graph_.detach().numpy()
            expl = expl_.detach().numpy()
            ground_truth = self.dataset_loader.edge_ground_truth[idx]

            for edge_idx in range(0, expl.shape[0]):  # Consider every edge in the ground truth
                edge_ = graph.T[edge_idx]
                if edge_[0] == edge_[1]:  # We dont consider self loops for our evaluation (Needed for Ba2Motif)
                    continue
                # Retrieve predictions and ground truth
                y_pred.append(expl[edge_idx])
                y_true.append(ground_truth[edge_idx])
        return y_true, y_pred

    def evaluate_auc(self, y_true, y_pred):
        return roc_auc_score(y_true, y_pred)

    def evaluate_rdd(self, ground_truth, explanation):
        """
        :param: explanation: edge weight vector
        :param: ground_truth: binary/float edge ground truth
        evaluate the reversed distribution distance from explanation to ground truth by f(x_{N})= 1 - root(x_{i}^{2})/N
        :return: float
        """
        assert len(explanation) == len(ground_truth)
        sum_d = 0.0
        for i in range(len(explanation)):
            sum_d += (explanation[i] - ground_truth[i]) ** 2
        return 1 - (sum_d / len(explanation)) ** 0.5

    def evaluation_auc_(self, explanations):
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
