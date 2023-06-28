import torch
import numpy as np
from utils.training import GNNTrainer

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

_explainer = 'GNNE'


def train_(dataset, model):
    trainer = GNNTrainer(dataset, model)
    # trainer.plot()
    trainer.train()


# train_('bareg1', 'GraphGCN')
# train_('bareg2', 'GraphGCN')
# train_('bareg1', 'RegGCN')
# train_('bareg2', 'RegGCN')
# train_('crippen', 'GraphGCN')
# train_('triangles_small', 'GraphGCN')

train_('crippen', 'GAT')
train_('bareg1', 'GAT')
train_('bareg2', 'GAT')
train_('triangles', 'GAT')
