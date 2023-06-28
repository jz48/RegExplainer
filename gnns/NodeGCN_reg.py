import torch
from torch.nn import ReLU, Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool
from torch_geometric.nn.dense import DenseGCNConv


class NodeGCN(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    """
    def __init__(self, num_features, num_classes):
        super(NodeGCN, self).__init__()
        self.embedding_size = 20 * 3
        self.conv1 = GCNConv(num_features, 20)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(20, 20)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(20, 20)
        self.relu3 = ReLU()
        self.lin = Linear(3*20, num_classes)

    def forward(self, x, edge_index, edge_weights=None):
        input_lin = self.embedding(x, edge_index, edge_weights)
        final = self.lin(input_lin)
        return final, input_lin

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))
        stack = []

        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)  # this is not used in PGExplainer
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  # this is not used in PGExplainer
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)  # this is not used in PGExplainer
        out3 = self.relu3(out3)
        stack.append(out3)

        input_lin = torch.cat(stack, dim=1)

        return input_lin


class GraphGCN(torch.nn.Module):
    """
    A graph clasification model for graphs decribed in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    In between the GCN outputs and linear layers are pooling operations in both mean and max.
    """
    def __init__(self, num_features, num_classes):
        super(GraphGCN, self).__init__()
        self.embedding_size = 20
        self.conv1 = GCNConv(num_features, 20)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(20, 20)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(20, 20)
        self.relu3 = ReLU()
        self.lin = Linear(self.embedding_size * 2, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None: # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)
        embed = self.embedding(x, edge_index, edge_weights)

        out1 = global_max_pool(embed, batch)
        out2 = global_mean_pool(embed, batch)
        input_lin = torch.cat([out1, out2], dim=-1)

        out = self.lin(input_lin)
        return out, input_lin

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))
        stack = []

        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)
        out3 = self.relu3(out3)

        input_lin = out3

        return input_lin


class GraphGCN2(torch.nn.Module):
    """
    A graph clasification model for graphs decribed in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    In between the GCN outputs and linear layers are pooling operations in both mean and max.
    """
    def __init__(self, num_features, num_classes):
        super(GraphGCN2, self).__init__()
        self.embedding_size = 20
        self.conv1 = GCNConv(num_features, 20)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(20, 20)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(20, 20)
        self.relu3 = ReLU()
        self.lin = Linear(self.embedding_size, num_classes)
        self.lin2 = Linear(num_classes, 1)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)
        embed = self.embedding(x, edge_index, edge_weights)

        # out1 = global_max_pool(embed, batch)
        # out2 = global_mean_pool(embed, batch)

        out1 = global_max_pool(embed, batch)
        out2 = global_mean_pool(embed, batch)

        input_lin = self.lin(embed)

        out = self.lin2(input_lin)
        return out, input_lin

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))
        stack = []

        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)
        out3 = self.relu3(out3)

        input_lin = out3

        return input_lin

    def loss(self, pred, score):
        return F.mse_loss(pred, score)

    def mape(self, y_pred, y):
        # print(y_pred, y)
        e = torch.abs(y.view_as(y_pred) - y_pred) / torch.abs(y.view_as(y_pred))
        # print(e)
        return 100.0 * e


class RegGNN(torch.nn.Module):
    '''Regression using a DenseGCNConv layer from pytorch geometric.
       Layers in this model are identical to GCNConv.
    '''

    def __init__(self, num_nodes, feature_dim, hidden_dim, num_class, dropout):
        super(RegGNN, self).__init__()

        self.gc1 = DenseGCNConv(feature_dim, hidden_dim)
        self.gc2 = DenseGCNConv(hidden_dim, num_class)
        self.dropout = dropout
        self.LinearLayer = torch.nn.Linear(num_nodes, 1)
        self.Lin2 = torch.nn.Linear(num_class, 1)

    def forward(self, x, edge_index, batch=None):
        # edge_index should be adj matrix
        # print(x)
        # print(edge_index)
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
        return x, out

    def loss(self, pred, score):
        return F.mse_loss(pred, score)

    def mape(self, y_pred, y):
        # print(y_pred, y)
        e = torch.abs(y.view_as(y_pred) - y_pred) / torch.abs(y.view_as(y_pred))
        # print(e)
        return 100.0 * e


class RegGraphGCN(torch.nn.Module):
    """
    A graph clasification model for graphs decribed in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    In between the GCN outputs and linear layers are pooling operations in both mean and max.
    """
    def __init__(self, num_features, num_classes):
        super(RegGraphGCN, self).__init__()
        self.embedding_size = 20
        self.conv1 = GCNConv(num_features, 20)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(20, 20)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(20, 20)
        self.relu3 = ReLU()
        self.lin = Linear(self.embedding_size * 2, num_classes)
        self.lin2 = Linear(num_classes, 1)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)
        embed = self.embedding(x, edge_index, edge_weights)

        out1 = global_max_pool(embed, batch)
        out2 = global_mean_pool(embed, batch)
        input_lin = torch.cat([out1, out2], dim=-1)

        out = self.lin(input_lin)
        out = self.lin2(out)
        return out, input_lin

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))
        stack = []

        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)
        out3 = self.relu3(out3)

        input_lin = out3

        return input_lin

    def loss(self, pred, score):
        return F.mse_loss(pred, score)

    def mape(self, y_pred, y):
        # print(y_pred, y)
        e = torch.abs(y.view_as(y_pred) - y_pred) / torch.abs(y.view_as(y_pred))
        # print(e)
        return 100.0 * e