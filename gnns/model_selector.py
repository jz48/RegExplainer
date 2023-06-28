import shutil

import torch
import os

from gnns.DenseRegGCN_reg import RegGCN as RegDenseGCN
from gnns.GraphGCN_reg import RegGraphGCN
from gnns.GraphGCN_classify import GraphGCN as GraphGCN_cls
from gnns.GraphGCN_classify import NodeGCN as NodeGCN_cls
from gnns.GAT_reg import GAT


def string_to_model(paper, dataset):
    """
    Given a paper and a dataset return the cooresponding neural model needed for training.
    :param paper: the paper who's classification model we want to use.
    :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    :returns: torch.nn.module models
    """
    print(paper)
    if paper == "GNN":
        if dataset == "bareg1":
            return GraphGCN(10, 200)
        elif dataset == "bareg2":
            return GraphGCN(10, 20)
        elif dataset == "mutag":
            return GraphGCN(14, 2)
        else:
            raise NotImplementedError
    elif paper == "REG":
        if dataset == "bareg1":
            return RegGNN(25, 10, 20, 200, 0.2)
        elif dataset == "bareg2":
            return RegGNN(120, 10, 20, 20, 0.2)
        elif dataset == "mutag":
            return RegGNN(14, 2)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def model_selector_(paper, dataset, device, pretrained=True, return_checkpoint=False):
    """
    Given a paper and dataset loads accociated model.
    :param paper: the paper who's classification model we want to use.
    :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    :param pretrained: whter to return a pre-trained model or not.
    :param return_checkpoint: wheter to return the dict contining the models parameters or not.
    :returns: torch.nn.module models and optionallly a dict containing it's parameters.
    """
    model = string_to_model(paper, dataset)
    model.device = device
    if pretrained:
        path = get_pretrained_path(paper, dataset)
        print(type(model))
        print('load model from path: ', path)
        checkpoint = torch.load(path)
        renamed_state_dict = {}
        for key in checkpoint['model_state_dict']:
            print(key)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            for key in checkpoint['model_state_dict']:
                if key.startswith('conv') and key.endswith('weight'):
                    new_key = key[:5] + key[-7:]
                    renamed_state_dict[new_key] = (checkpoint['model_state_dict'][key])
                    # renamed_state_dict[new_key] = (checkpoint['model_state_dict'][key]).T
                else:
                    renamed_state_dict[key] = checkpoint['model_state_dict'][key]
            model.load_state_dict(renamed_state_dict)
        # model.load_state_dict(checkpoint['model_state_dict'])
        print(
            f"This model obtained: Train Acc: {checkpoint['train_acc']:.4f}, Val Acc: {checkpoint['val_acc']:.4f}, Test Acc: {checkpoint['test_acc']:.4f}.")
        if return_checkpoint:
            return model, checkpoint
    return model


class ModelSelector:
    def __init__(self, model_name, dataset_name, load_pretrain=False):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.load_pretrain = load_pretrain
        self.model = self.select_gnn_model()
        if load_pretrain:
            self.load_best_model(-1, True)
        pass

    def select_gnn_model(self):
        if self.model_name == 'GraphGCN':
            if self.dataset_name == 'bareg1':
                return RegGraphGCN('train', 10, 200)
            elif self.dataset_name == 'bareg2':
                return RegGraphGCN('train', 10, 21)
            elif self.dataset_name == 'crippen':
                return RegGraphGCN('train', 14, 100)
            elif self.dataset_name in ['triangles', 'triangles_small']:
                return RegGraphGCN('train', 10, 100)
            elif self.dataset_name == 'ba2motif':
                return GraphGCN_cls('train', 10, 2)
        elif self.model_name == 'GAT':
            if self.dataset_name == 'bareg1':
                return GAT('train', 10, 25)
            elif self.dataset_name == 'bareg2':
                return GAT('train', 10, 120)
            elif self.dataset_name == 'crippen':
                return GAT('train', 14, 120)
            elif self.dataset_name in ['triangles', 'triangles_small']:
                return GAT('train', 10, 30)
        elif self.model_name == 'RegGCN':
            if self.dataset_name == 'bareg1':
                return RegDenseGCN('train', 25, 10, 20, 200, 0.2)
            elif self.dataset_name == 'bareg2':
                return RegDenseGCN('train', 120, 10, 20, 21, 0.2)
        assert 0

    def store_checkpoint(self, model_name, dataset, model, train_acc, val_acc, epoch=-1):
        """
        Store the model weights at a predifined location.
        :param model_name: str, the model_name
        :param dataset: str, the dataset
        :param model: the model who's parameters we wish to save
        :param train_acc: training accuracy obtained by the model
        :param val_acc: validation accuracy obtained by the model
        :param test_acc: test accuracy obtained by the model
        :param epoch: the current epoch of the training process
        :retunrs: None
        """
        save_dir = f"./data/checkpoints/{model_name}/{dataset}"
        checkpoint = {'model_state_dict': model.state_dict(),
                      'train_acc': train_acc,
                      'val_acc': val_acc}
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        elif epoch == 0:
            shutil.rmtree(save_dir)
            os.makedirs(save_dir)

        if epoch == -1:
            torch.save(checkpoint, os.path.join(save_dir, f"best_model"))
        else:
            torch.save(checkpoint, os.path.join(save_dir, f"model_{epoch}"))

    def store_best_checkpoint(self, model_name, dataset, model, train_acc, val_acc, epoch=-1):
        """
        Store the model weights at a predifined location.
        :param model_name: str, the model_name
        :param dataset: str, the dataset
        :param model: the model who's parameters we wish to save
        :param train_acc: training accuracy obtained by the model
        :param val_acc: validation accuracy obtained by the model
        :param test_acc: test accuracy obtained by the model
        :param epoch: the current epoch of the training process
        :retunrs: None
        """
        save_dir = f"./data/checkpoints/{model_name}/{dataset}"
        shutil.copy(os.path.join(save_dir, f"model_{epoch}"), os.path.join(save_dir, f"best_model"))

    def load_best_model(self, best_epoch, eval_enabled):
        """
        Load the model parameters from a checkpoint into a model
        :param best_epoch: the epoch which obtained the best result. use -1 to chose the "best model"
        :param eval_enabled: wheater to activate evaluation mode on the model or not
        :return: model with pramaters taken from the checkpoint
        """
        print(best_epoch)

        path = '/Users/jiaxingzhang/Desktop/Lab/XAI_GNN/XAIG_REG/data/checkpoints'
        if best_epoch == -1:
            path = os.path.join(path, self.model_name, self.dataset_name, 'best_model')
            # checkpoint = torch.load(f"./data/checkpoints/{self.model_name}/{self.dataset_name}/best_model")
            checkpoint = torch.load(path)
        else:
            path = os.path.join(path, self.model_name, self.dataset_name, 'model_'+str(best_epoch))
            # checkpoint = torch.load(f"./data/checkpoints/{self.model_name}/{self.dataset_name}/model_{best_epoch}")
            checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if eval_enabled:
            self.model.eval()

