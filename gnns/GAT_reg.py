import torch
from torch.nn import ReLU, Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool, GATConv
from torch_geometric.nn.dense import DenseGCNConv
from gnns.BaseModel import BaseModel


class GAT(BaseModel):
    """
    A graph regression model
    This model consists of 3 stacked GCN layers followed by a linear layer.
    In between the GCN outputs and linear layers are pooling operations in both mean and max.
    """
    def __init__(self, training, num_features, num_nodes=25, num_classes=1):
        super().__init__(training)

        self.hid = 8
        self.in_head = 8
        self.out_head = 1
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.conv1 = GATConv(num_features, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid * self.in_head, num_classes, concat=False,
                             heads=self.out_head, dropout=0.6)

        self.lin = Linear(self.num_nodes, 1)

    def forward(self, x, edge_index, batch, edge_weights=None):
        a = x.shape
        if a[0] != self.num_nodes:
            x_ = torch.zeros(self.num_nodes, self.num_features)
            x_[:a[0], :] = x
            x = x_
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x, attention_weight = self.conv2(x, edge_index, return_attention_weights=True)
        x = torch.transpose(x, 0, 1)
        x = self.lin(x)
        return x
