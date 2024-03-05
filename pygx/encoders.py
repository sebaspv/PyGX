import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False)
        self.conv2 = GCNConv(2 * out_channels, 2 * out_channels, cached=False)
        self.mu = GCNConv(2 * out_channels, out_channels, cached=False)
        self.logstd = GCNConv(2 * out_channels, out_channels, cached=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index).relu()
        return self.mu(x, edge_index), self.logstd(x, edge_index)
