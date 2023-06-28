from abc import ABC, abstractmethod
import torch
import random
from sklearn.metrics.pairwise import cosine_similarity
from utils.contrastive_loss import ContrastiveLoss


class BaseExplainer(ABC):
    def __init__(self, model_to_explain, graphs, features, task, model_type, loss_type):
        self.model_to_explain = model_to_explain
        self.graphs = graphs
        self.features = features
        self.task = task
        self.model_type = model_type  # reg or cls
        self.loss_type = loss_type  # ib(information bottleneck), inpl(inner product loss) or ctl(contrastive loss)
        self.Contrastive_loss = ContrastiveLoss()
        self.beta = 1.0
        if loss_type == 'ib':
            if self.model_type == 'cls':
                self.loss_name = 'ibcls'
            elif self.model_type == 'reg':
                self.loss_name = 'ibreg'
        elif loss_type == 'inpl':
            self.loss_name = 'inpl'
        elif loss_type == 'ctl':
            self.loss_name = 'ctl'
        else:
            assert 0

    @abstractmethod
    def prepare(self, args):
        """Prepars the explanation method for explaining.
        Can for example be used to train the method"""
        pass

    @abstractmethod
    def explain(self, index):
        """
        Main method for explaining samples
        :param index: index of node/graph in self.graphs
        :return: explanation for sample
        """
        pass

    @abstractmethod
    def serialize_configuration(self):
        """
        Main method for output configuration
        :return: a text sequence
        """
        pass

    def build_neighbors_sim_score(self, indices):
        self.neighbor_pairs = {}
        self.neighbor_matrix = {idx: {} for idx in indices}
        # for index in range(len(self.graphs)):
        for index in indices:
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
                    self.neighbor_matrix[key][key2] = 1.0
                    continue
                if key2 < key:
                    self.neighbor_matrix[key][key2] = self.neighbor_matrix[key2][key]
                    continue
                cos_sim = cosine_similarity(self.neighbor_pairs[key][0], self.neighbor_pairs[key2][0])
                self.neighbor_matrix[key][key2] = cos_sim.tolist()[0][0]
        return

    def sample_index(self, index_, if_reverse):
        tmp = self.neighbor_matrix[index_]
        indices = []
        weight_list = []
        for _, key in enumerate(tmp):
            indices.append(key)
            if if_reverse:
                weight_list.append(1.0-tmp[key])
            else:
                weight_list.append(tmp[key])
        while True:
            target = random.choices(indices, weights=weight_list, k=1)[0]
            if target != index_:
                break
        return target

    def loss(self, masked_pred, original_pred, edge_mask, reg_coefs):
        if self.loss_type == 'inpl':
            return self.inner_product_loss(masked_pred, original_pred)
        elif self.loss_type == 'ib':
            return self._loss(masked_pred, original_pred, edge_mask, reg_coefs)
        elif self.loss_type == 'ctl':
            return self.contrastive_loss(masked_pred, original_pred, edge_mask, reg_coefs)

    def _loss(self, masked_pred, original_pred, edge_mask, reg_coefs):
        """
        Returns the loss score based on the given mask.
        :param masked_pred: Prediction based on the current explanation
        :param original_pred: Predicion based on the original graph
        :param edge_mask: Current explanaiton
        :param reg_coefs: regularization coefficients
        :return: loss
        """
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1]
        EPS = 1e-15

        # Regularization losses
        mask = torch.sigmoid(edge_mask)
        # mask = edge_mask
        size_loss = torch.sum(mask) * size_reg

        # here is a question about this loss, it seems doesn't minimize at x=0/1
        mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        # Explanation loss
        if self.model_type == 'cls':
            pred_label = original_pred.argmax(dim=-1).detach()
            cce_loss = torch.nn.functional.cross_entropy(masked_pred, pred_label)
            self.loss_data = {
                'loss/cce_loss': cce_loss.detach().numpy(),
                'loss/size_loss': size_loss.detach().numpy(),
                'loss/mask_ent_loss': mask_ent_loss.detach().numpy(),
            }
            return cce_loss + size_loss + mask_ent_loss
        elif self.model_type == 'reg':
            if 1:
                mse_loss = torch.nn.functional.mse_loss(masked_pred, original_pred)
            else:
                y = original_pred
                y_pred = masked_pred
                mse_loss = 100 * torch.abs(y.view_as(y_pred) - y_pred) / torch.abs(y.view_as(y_pred))
            self.loss_data = {
                'loss/mse_loss': mse_loss.detach().numpy(),
                'loss/size_loss': size_loss.detach().numpy(),
                'loss/mask_ent_loss': mask_ent_loss.detach().numpy(),
            }
            return mse_loss + size_loss + mask_ent_loss
        else:
            assert 0

    def gib_without_mse(self, masked_pred, original_pred, edge_mask, reg_coefs):
        """
        Returns the loss score based on the given mask.
        :param masked_pred: Prediction based on the current explanation
        :param original_pred: Predicion based on the original graph
        :param edge_mask: Current explanaiton
        :param reg_coefs: regularization coefficients
        :return: loss
        """
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1]
        EPS = 1e-15

        # Regularization losses
        mask = torch.sigmoid(edge_mask)
        # mask = edge_mask
        size_loss = torch.sum(mask) * size_reg

        # here is a question about this loss, it seems doesn't minimize at x=0/1
        # mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
        # mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)
        inpl = self.inner_product_loss(mask, None)
        # Explanation loss
        if self.model_type == 'cls':
            pred_label = original_pred.argmax(dim=-1).detach()
            cce_loss = torch.nn.functional.cross_entropy(masked_pred, pred_label)
            self.loss_data = {
                'loss/cce_loss': cce_loss.detach().numpy(),
                'loss/size_loss': size_loss.detach().numpy(),
                'loss/inpl': inpl.detach().numpy(),
            }
            return size_loss + inpl
        elif self.model_type == 'reg':
            if 1:
                mse_loss = torch.nn.functional.mse_loss(masked_pred, original_pred)
            else:
                y = original_pred
                y_pred = masked_pred
                mse_loss = 100 * torch.abs(y.view_as(y_pred) - y_pred) / torch.abs(y.view_as(y_pred))
            self.loss_data = {
                'loss/mse_loss': mse_loss.detach().numpy(),
                'loss/size_loss': size_loss.detach().numpy(),
                'loss/inpl': inpl.detach().numpy(),
            }
            return size_loss + inpl
        else:
            assert 0

    def contrastive_loss(self, masked_pred, original_pred, edge_mask, reg_coefs):
        """
        Returns the loss score based on the given mask.
        :param masked_pred: Prediction based on the current explanation
        :param original_pred: Predicion based on the original graph
        :param edge_mask: Current explanaiton
        :param reg_coefs: regularization coefficients
        :return: loss
        """
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1]
        EPS = 1e-15
        beta = torch.tensor(self.beta, requires_grad=False)
        # Regularization losses
        mask = torch.sigmoid(edge_mask)
        # mask = edge_mask
        size_loss = torch.sum(mask) * size_reg

        # here is a question about this loss, it seems doesn't minimize at x=0/1
        # mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
        # mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)
        inpl = self.inner_product_loss(mask, None)
        # Explanation loss
        if self.model_type == 'cls':
            pred_label = original_pred.argmax(dim=-1).detach()
            cce_loss = torch.nn.functional.cross_entropy(masked_pred, pred_label)
            self.loss_data = {
                'loss/cce_loss': cce_loss.detach().numpy(),
                'loss/size_loss': size_loss.detach().numpy(),
                'loss/inpl': inpl.detach().numpy(),
            }
            return beta * cce_loss + size_loss + inpl
        elif self.model_type == 'reg':
            if 1:
                mse_loss = torch.nn.functional.mse_loss(masked_pred, original_pred)
            else:
                y = original_pred
                y_pred = masked_pred
                mse_loss = 100 * torch.abs(y.view_as(y_pred) - y_pred) / torch.abs(y.view_as(y_pred))
            self.loss_data = {
                'loss/mse_loss': mse_loss.detach().numpy(),
                'loss/size_loss': size_loss.detach().numpy(),
                'loss/inpl': inpl.detach().numpy(),
            }
            return beta * mse_loss + size_loss + inpl
        else:
            assert 0

    def inner_cce_loss(self, masked_pred, original_pred):
        inner_product_loss = -torch.log(torch.sigmoid(masked_pred) * torch.sigmoid(original_pred)).sum()
        self.loss_data = {
            'loss/inner_product_loss': inner_product_loss.detach().numpy(),
        }
        return inner_product_loss

    def inner_product_loss(self, masked_pred, original_pred):
        inpl = masked_pred * masked_pred
        inpl = torch.sigmoid(inpl)
        inpl = -torch.log(inpl)
        inner_product_loss = inpl.sum()
        self.loss_data = {
            'loss/inner_product_loss': inner_product_loss.detach().numpy(),
        }
        return inner_product_loss

    def cross_inner_product_loss(self, masked_pred, original_pred):
        inpl = masked_pred * original_pred
        inpl = torch.sigmoid(inpl)
        inpl = -torch.log(inpl)
        inner_product_loss = inpl.sum()
        self.loss_data = {
            'loss/inner_product_loss': inner_product_loss.detach().numpy(),
        }
        return inner_product_loss
