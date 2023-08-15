import argparse
import os
import pandas as pd
import os.path as osp
import torch
import numpy as np
import warnings
import torch_geometric.transforms as T
from torch_geometric.seed import seed_everything as th_seed
from torch_geometric.datasets import Planetoid,WebKB,Actor,WikipediaNetwork, LINKXDataset
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import to_dense_adj, to_undirected
from models import *
from utils import *
warnings.filterwarnings("ignore")
torch.manual_seed(1234)
np.random.seed(1234)
################### Arguments parameters ###################################
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    default="texas",
    choices=["texas","wisconsin","actor","cornell","squirrel","chamaleon","cora","citeseer","pubmed","penn94"],
    help="You can choose between texas, wisconsin, actor, cornell, squirrel, chamaleon, cora, citeseer, pubmed",
)
parser.add_argument(
    "--cuda",
    default="cpu",
    choices=["cuda:0","cuda:1","cpu"],
    help="You can choose between cuda:0, cuda:1, cpu",
)
parser.add_argument(
        "--hidden_channels", type=int, default=128, help="Hidden channels for the unsupervised model"
)
parser.add_argument(
        "--lr", type=float, default=0.0001, help="Outer learning rate of model"
    )
parser.add_argument(
    "--model",
    default="LinkWIRE",
    choices=["VAE","GAE","GNAE","VGNAE","LinkWIRE"],
    help="List of models to choose from: VAE, GAE, GNAE, VGNAE, LinkWIRE",
)
parser.add_argument(
        "--epochs", type=int, default=100, help="Epochs for the model"
    )

parser.add_argument(
        "--num_centers", type=int, default=3, help="Number of eigenvectors"
)
parser.add_argument(
    "--dropout", type=float, default=0.35, help="Dropout probability"
)
args = parser.parse_args()
################### Importing the dataset ###################################
if args.dataset == "texas":
    dataset = WebKB(root='./data',name='texas')
    data = dataset[0]
elif args.dataset == "wisconsin":
    dataset = WebKB(root='./data',name='wisconsin')
    data = dataset[0]
elif args.dataset == "actor":
    dataset  = Actor(root='./data')
    dataset.name = "film"
    data = dataset[0]
elif args.dataset == "cornell":
    dataset = WebKB(root='./data',name='cornell')
    data = dataset[0]
elif args.dataset == "squirrel":
    dataset = WikipediaNetwork(root='./data',name='squirrel')
    data = dataset[0]    
elif args.dataset == "chamaleon":
    dataset = WikipediaNetwork(root='./data',name='chameleon')
    data = dataset[0]
elif args.dataset == "cora":
    dataset = Planetoid(root='./data',name='cora')
    data = dataset[0]
elif args.dataset == "citeseer":
    dataset = Planetoid(root='./data',name='citeseer')
    data = dataset[0]
elif args.dataset == "pubmed":
    dataset = Planetoid(root='./data',name='pubmed')
    data = dataset[0]
elif args.dataset == "penn94":
    dataset = LINKXDataset(root='./data',name='penn94')
    data = dataset[0]
print()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
print()
print(data) 
print('===========================================================================================================')
adj = to_dense_adj(data.edge_index)[0] # Convert the sparse adjacency matrix to a dense adjacency matrix
print("Shape of the adjacency matrix: ",adj.shape)
print("Shape of the new adjacency matrix: ",adj.shape)
#Let's get statistics about the graph adjacency matrix
print("Number of nodes: ",adj.shape[0])
print("Number of edges: ",round(adj.sum().item()))
print("Density: ",round((adj.sum()/(adj.shape[0]*adj.shape[0])).item(),4))
print("Maximum node degree: ",round(adj.sum(axis=1).max().item()))
print("Minimum node degree: ",round(adj.sum(axis=1).min().item()))
print("Average node degree: ",round(adj.sum(axis=1).mean().item()))
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
if data.is_undirected() == False:
    adj = to_undirected(data.edge_index)
    adj = to_dense_adj(adj)[0]
print('===========================================================================================================')        
################### CUDA ###################################
device = torch.device(args.cuda)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data = data.to(device)   
adj = adj.to(device)
print("Device: ",device)
################### Transformations ###################################
# Now let's create 10 splits of the dataset
print("Creating 10 splits of the dataset...")
results = []
seeds = [9234, 8579, 9012, 3966, 7890, 2721, 6321, 2023, 3059, 7280]
for i,seed in enumerate(seeds):
    th_seed(seed)
    transform = T.Compose([
    #T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1),
    # Also parse to undirected
    T.ToUndirected()
    ])
    data = transform(dataset[0])
    train_data, val_data, test_data = data
    # parse the edge_index to undirected
    #train_data.edge_index = to_undirected(train_data.edge_index)
    #val_data.edge_index = to_undirected(val_data.edge_index)
    #test_data.edge_index = to_undirected(test_data.edge_index)
    best_val_auc = final_test_auc = 0

    model = LinkWIRE(in_channels=dataset.num_features, hidden_channels=args.hidden_channels,num_centers = args.num_centers,adj_dim = adj.shape[0],out_channels=dataset.num_classes, drop_out = args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_val_auc = final_test_auc = 0
    final_test_ap = 0
    for epoch in range(1, args.epochs+1):
        loss,auc_train,ap_train = train(model,train_data,optimizer,criterion,adj)
        val_auc,val_ap = test(model,val_data,adj)
        test_auc,test_ap = test(model,test_data,adj)
        if test_auc >= best_val_auc:
            best_val_auc = test_auc        
            final_test_auc = test_auc
            final_test_ap = test_ap
        #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc_train:.4f}, AP: {ap_train:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}, Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}')
    print("The results for the split: ",i," is: AUC: ",final_test_auc," AP: ",final_test_ap)
    results.append([final_test_auc,final_test_ap])
print("Done!")
print("The result for the dataset: ",args.dataset," is: AUC: ",np.mean(np.array(results)[:,0])*100,"+-",np.std(np.array(results)[:,0])*100)
print("The result for the dataset: ",args.dataset," is: AP: ",np.mean(np.array(results)[:,1])*100,"+-",np.std(np.array(results)[:,1])*100)

print("Arguments: ",args)
# Now we check if it is created a csv with the configuration and the results
if os.path.isfile('results.csv'):
    # If the file exists, then we append the configuration and the results
    # The columns are: dataset, model, hidden_channels, lr, epochs, num_centers, AUC, AP
    res = pd.read_csv('results.csv')
    # Check if the configuration is already in the csv
    if res[(res['dataset'] == args.dataset) & (res['model'] == args.model) & (res['hidden_channels'] == args.hidden_channels) & (res['lr'] == args.lr) & (res['epochs'] == args.epochs) & (res['num_centers'] == args.num_centers)].shape[0] == 0:
        # If the configuration is not in the csv, then we append it
        res = res.append({'dataset': args.dataset, 'model': args.model, 'hidden_channels': args.hidden_channels, 'lr': args.lr, 'epochs': args.epochs, 'num_centers': args.num_centers, 'AUC': np.mean(np.array(results)[:,0])*100, 'AP': np.mean(np.array(results)[:,1])*100}, ignore_index=True)
        res.to_csv('results.csv', index=False)
    res.to_csv('results.csv', index=False)
else:
    # If the file does not exist, then we create it and append the configuration and the results
    res = pd.DataFrame(columns=['dataset', 'model', 'hidden_channels', 'lr','dropout', 'epochs', 'num_centers', 'AUC', 'AP',])
    res = res.append({'dataset': args.dataset, 'model': args.model, 'hidden_channels': args.hidden_channels, 'lr': args.lr,'dropout':args.dropout, 'epochs': args.epochs, 'num_centers': args.num_centers, 'AUC': np.round(np.mean(np.array(results)[:,0])*100,2), 'AP': np.round(np.mean(np.array(results)[:,1])*100,2)}, ignore_index=True)
    res.to_csv('results.csv', index=False)