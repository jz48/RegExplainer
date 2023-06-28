import torch
import numpy as np
from utils.training import GNNTrainer

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)


def statistic_datasets():
    datasets = ['bareg1', 'bareg2', 'triangles', 'crippen']
    model = 'GraphGCN'
    for dataset in datasets:
        num_nodes = []
        trainer = GNNTrainer(dataset, model)
        feats = trainer.dataset_loader.features
        for feat in feats:
            num_node = feat.shape[0]
            num_nodes.append(num_node)
        num_nodes = np.array(num_nodes)
        print(dataset, num_nodes.mean(), num_nodes.max())


statistic_datasets()

