import argparse
import os.path as osp

import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork
from torch_geometric.nn import GAE, VGAE, GCNConv, DenseGCNConv, ARGVA
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_undirected
from torch_geometric.seed import seed_everything as th_seed

parser = argparse.ArgumentParser()
parser.add_argument('--variational', action='store_true')
parser.add_argument('--adversarial', action='store_true')
parser.add_argument('--linear', action='store_true')
parser.add_argument(
    "--dataset",
    default="texas",
    choices=["texas","wisconsin","actor","cornell","squirrel","chamaleon","cora","citeseer","pubmed","pen94"],
    help="You can choose between texas, wisconsin, actor, cornell, squirrel, chamaleon, cora, citeseer, pubmed",
)
parser.add_argument('--epochs', type=int, default=2000)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logstd = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.lin1(x).relu()
        x = self.lin2(x).relu()
        return self.lin3(x)


bests_auc = []
bests_ap = []
for seed in [9234, 8579, 9012, 3966, 7890, 2721, 6321, 9012, 3059, 7280]:
    th_seed(seed)
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                        split_labels=True, add_negative_train_samples=False),
    ])
    ################### Importing the dataset ###################################
    if args.dataset == "texas":
        dataset = WebKB(root='./data',name='texas',transform=transform)
        data = dataset[0]
        print("Original dataset")
        print( WebKB(root='./data',name='texas')[0])
    elif args.dataset == "wisconsin":
        dataset = WebKB(root='./data',name='wisconsin',transform=transform)
        data = dataset[0]
    elif args.dataset == "actor":
        dataset  = Actor(root='./data',transform=transform)
        dataset.name = "film"
        data = dataset[0]
    elif args.dataset == "cornell":
        dataset = WebKB(root='./data',name='cornell',transform=transform)
        data = dataset[0]
    elif args.dataset == "squirrel":
        dataset = WikipediaNetwork(root='./data',name='squirrel',transform=transform)
        data = dataset[0]    
    elif args.dataset == "chamaleon":
        dataset = WikipediaNetwork(root='./data',name='chameleon',transform=transform)
        data = dataset[0]
    elif args.dataset == "cora":
        dataset = Planetoid(root='./data',name='cora',transform=transform)
        data = dataset[0]
    elif args.dataset == "citeseer":
        dataset = Planetoid(root='./data',name='citeseer',transform=transform)
        data = dataset[0]
    elif args.dataset == "pubmed":
        dataset = Planetoid(root='./data',name='pubmed',transform=transform)
        data = dataset[0]
    
        
    train_data, val_data, test_data = dataset[0]

    in_channels, out_channels = dataset.num_features, 16
    data = dataset
    adj = to_dense_adj(data.edge_index)[0].to(device)
    num_of_nodes = adj.size(0)


    if not args.variational and not args.linear and args.adversarial == False:
        model = GAE(GCNEncoder(in_channels, out_channels))
    elif not args.variational and args.linear and args.adversarial == False:
        model = GAE(LinearEncoder(in_channels, out_channels))
    elif args.variational and not args.linear:
        model = VGAE(VariationalGCNEncoder(in_channels, out_channels))
    elif args.variational and args.linear and args.adversarial == False:
        model = VGAE(VariationalLinearEncoder(in_channels, out_channels))
    if args.adversarial:
        encoder = Encoder(train_data.num_features, hidden_channels=32, out_channels=32)
        discriminator = Discriminator(in_channels=32, hidden_channels=64,
                              out_channels=32)
        model = ARGVA(encoder, discriminator).to(device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    print(train_data)
    print(val_data)
    print(test_data)
    #exit()
    def train(train_data):
        model.train()
        optimizer.zero_grad()
        if train_data.is_undirected() == False:
            train_data = train_data.to_undirected()
        z = model.encode(train_data.x, train_data.edge_index)
        
        loss = model.recon_loss(z, train_data.pos_edge_label_index) 
        if args.variational:
            loss = loss + (1 / train_data.num_nodes) * model.kl_loss() 
        loss.backward()
        optimizer.step()
        return float(loss)


    @torch.no_grad()
    def test(data):
        model.eval()
        if data.is_undirected() == False:
            data = data.to_undirected()
        z = model.encode(data.x, data.edge_index)
        return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
    best_val_auc = final_test_auc = 0
    for epoch in range(1, args.epochs + 1):
        loss = train(train_data)
        auc_val, ap_val = test(val_data)
        auc_test, ap_test = test(test_data)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
            f'AUC_val: {auc_val:.4f}, AP_val: {ap_val:.4f}, '
            f'AUC_test: {auc_test:.4f}, AP_test: {ap_test:.4f}')
        if auc_val > best_val_auc:
            best_val_auc = auc_val
            final_test_auc = auc_test
            final_test_ap = ap_test
    print(f'Best AUC: {final_test_auc:.4f}, Best AP: {final_test_ap:.4f}')
    bests_auc.append(final_test_auc)
    bests_ap.append(final_test_ap)
print("AUC with 10 splits: ",sum(bests_auc)/len(bests_auc)*100," +- ",np.std(bests_auc)*100)
print("AP with 10 splits: ",sum(bests_ap)/len(bests_ap)*100," +- ",np.std(bests_ap)*100)
print("Args: ",args)