import torch
import torch.nn.functional as F
from torch_geometric.nn.dense import DenseGCNConv
from gnns.BaseModel import BaseModel


class RegGCN(BaseModel):
    '''Regression using a DenseGCNConv layer from pytorch geometric.
       Layers in this model are identical to GCNConv.
    '''

    def __init__(self, training, num_nodes, feature_dim, hidden_dim, num_class, dropout):
        super().__init__(training)

        self.gc1 = DenseGCNConv(feature_dim, hidden_dim)
        self.gc2 = DenseGCNConv(hidden_dim, num_class)
        self.dropout = dropout
        self.LinearLayer = torch.nn.Linear(num_nodes, 1)
        self.Lin2 = torch.nn.Linear(num_class, 1)

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)
        # print(x.shape, edge_index.shape)
        x = self.gc1(x, edge_index)
        x = F.relu(x)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        # print(x.shape, edge_index.shape)
        x = torch.transpose(x, 2, 1)
        # print(x.shape)
        x = self.LinearLayer(x)
        # print(x.shape)
        out = torch.squeeze(x, 2)
        # print(x.shape)
        x = self.Lin2(out)
        # assert 0
        return x
