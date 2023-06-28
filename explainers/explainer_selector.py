import shutil

import torch
import os

from explainers.GNNExplainer import GNNExplainer
from explainers.MIXGNNExplainer import MixUpGNNExplainer, MixUpSFTGNNExplainer
from explainers.PGExplainer import PGExplainer
from explainers.MIXPGExplainer import MixUpPGExplainer
from explainers.MIXSFTPGExplainer import MixUpSFTPGExplainer
from explainers.CTLMIXSFTPGExplainer import CTLMixUpSFTPGExplainer
from explainers.CTLMIX_ALT_PGExplainer import CTLALTMixupPGExplainer
from explainers.NAMGNNExplainer import NAMGNNExplainer
from explainers.NAMPGExplainer import NAMPGExplainer
from explainers.CTLGNNExplainer import CTLGNNExplainer
from explainers.CTLPGExplainer import CTLPGExplainer
from explainers.GRAD_Explainer import GRADExplainer
from explainers.GAT_Explainer import GATExplainer
from utils.wandb_logger import WandbLogger
from explainers.CTLPGExplainer_no_mixup import CTLPGExplainer_no_mix
from explainers.CTLPGExplainer_no_contrastive import CTLPGExplainer_no_contrastive
from explainers.CTLMIXSFTPGExplainer_no_mixup import CTLMixUpSFTPGExplainer_no_mix
from explainers.CTLMIXSFTPGExplainer_no_contrastive import CTLMixUpSFTPGExplainer_no_cl
from explainers.CTLPGExplainer_no_mse import CTLPGExplainer_no_mse
from explainers.CTLMIXSFTPGExplainer_no_mse import CTLMixUpSFTPGExplainer_no_mse


class ExplainerSelector:
    def __init__(self, explainer_name, model_name, dataset_name, model_to_explain,
                 loss_type, graphs, features):
        self.explainer_name = explainer_name

        self.model_name = model_name
        if model_name in ['GraphGCN', 'GAT']:
            self.task = 'graph'
        else:
            self.task = 'node'

        self.dataset_name = dataset_name
        if dataset_name in ['bareg1', 'bareg2', 'bareg3', 'crippen', 'triangles', 'triangles_small']:
            self.model_type = 'reg'
        else:
            self.model_type = 'cls'

        self.model_to_explain = model_to_explain
        self.graphs = graphs
        self.features = features

        self.loss_type = loss_type
        self.explainer = self.select_explainer_model()

    def select_explainer_model(self):
        if self.explainer_name == 'GNNE':
            return GNNExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                self.loss_type)
        elif self.explainer_name == 'MIXGNNE':
            return MixUpGNNExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                     self.loss_type)
        elif self.explainer_name == 'NAMGNNE':
            return NAMGNNExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                   self.loss_type)
        elif self.explainer_name == 'CTLGNNE':
            return CTLGNNExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                   self.loss_type)
        elif self.explainer_name == 'PGE':
            return PGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                               self.loss_type)
        elif self.explainer_name == 'MIXPGE':
            return MixUpPGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                    self.loss_type)
        elif self.explainer_name == 'MIXSFTPGE':
            return MixUpSFTPGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                       self.loss_type)
        elif self.explainer_name == 'CTLPGE':
            return CTLPGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                  self.loss_type)
        elif self.explainer_name == 'CTLMIXSFTPGE':
            return CTLMixUpSFTPGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                          self.loss_type)
        elif self.explainer_name == 'CTLALTMIXPGE':
            return CTLALTMixupPGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                          self.loss_type)
        elif self.explainer_name == 'NAMPGE':
            return NAMPGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                  self.loss_type)
        elif self.explainer_name == 'GRAD':
            return GRADExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                 self.loss_type)
        elif self.explainer_name == 'GAT':
            return GATExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                self.loss_type)
        elif self.explainer_name == 'CTLPGE_no_mix':
            return CTLPGExplainer_no_mix(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                         self.loss_type)
        elif self.explainer_name == 'CTLPGE_no_cl':
            return CTLPGExplainer_no_contrastive(self.model_to_explain, self.graphs, self.features, self.task,
                                                 self.model_type, self.loss_type)
        elif self.explainer_name == 'CTLPGE_no_mse':
            return CTLPGExplainer_no_mse(self.model_to_explain, self.graphs, self.features, self.task,
                                         self.model_type, self.loss_type)
        elif self.explainer_name == 'CTLMIXSFTPGE_no_mix':
            return CTLMixUpSFTPGExplainer_no_mix(self.model_to_explain, self.graphs, self.features, self.task,
                                                 self.model_type, self.loss_type)
        elif self.explainer_name == 'CTLMIXSFTPGE_no_cl':
            return CTLMixUpSFTPGExplainer_no_cl(self.model_to_explain, self.graphs, self.features, self.task,
                                                self.model_type, self.loss_type)
        elif self.explainer_name == 'CTLMIXSFTPGE_no_mse':
            return CTLMixUpSFTPGExplainer_no_mse(self.model_to_explain, self.graphs, self.features, self.task,
                                                 self.model_type, self.loss_type)
        elif self.explainer_name == 'RegGCN':
            if self.dataset_name == 'bareg1':
                return RegDenseGCN('train', 25, 10, 20, 200, 0.2)
            elif self.dataset_name == 'bareg2':
                return RegDenseGCN('train', 120, 10, 20, 21, 0.2)
        assert 0
