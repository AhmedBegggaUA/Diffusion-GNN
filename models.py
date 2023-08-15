import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, DenseGCNConv,APPNP
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F
from pump import *

class LinkWIRE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,num_centers,adj_dim,out_channels,drop_out = 0.35):
            super(LinkWIRE, self).__init__()
            torch.manual_seed(1234)
            self.MLP = Linear(adj_dim, hidden_channels)
            self.MLP2 = Linear(hidden_channels, num_centers)
            #GNN
            self.pre_conv = GCNConv(in_channels, hidden_channels)
            self.convs = DenseGCNConv(in_channels,hidden_channels)
            self.convs2 = DenseGCNConv(hidden_channels, hidden_channels)
            self.convs3 = DenseGCNConv(hidden_channels, hidden_channels)
            self.convs4 = DenseGCNConv(hidden_channels, hidden_channels)
            #Aux
            self.drop_out = drop_out
            # Loss
            self.losses = 0
    def encode(self, x, edge_index,adj):
        empty_adj = torch.zeros_like(adj)
        empty_adj[edge_index[0],edge_index[1]] = 1
        s = self.MLP(empty_adj)
        s = self.MLP2(s)
        _, pump_loss, ortho_loss,distance_matrix= pump(empty_adj, s)
        distance_matrix = distance_matrix.squeeze(0)
        #print(x.type())
        #print(empty_adj.shape)
        z = self.convs(x,empty_adj.unsqueeze(0)).squeeze(0).relu() 
        z = F.dropout(z, p=self.drop_out, training=self.training)
        z = self.convs2(z,distance_matrix.unsqueeze(0)).squeeze(0)
        self.losses = pump_loss + ortho_loss
        return z
    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()