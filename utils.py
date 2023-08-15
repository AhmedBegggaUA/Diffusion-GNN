import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from torch_geometric.utils import negative_sampling
def train(model,train_data,optimizer,criterion,adj):
    #train_data.edge_label_index = train_data.pos_edge_label_index
    #train_data.edge_label = train_data.pos_edge_label
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x,train_data.edge_index,adj)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label) + model.losses
    loss.backward()
    optimizer.step()
    return loss, roc_auc_score(edge_label.cpu().numpy(), out.detach().cpu().numpy()),average_precision_score(edge_label.cpu().numpy(), out.detach().cpu().numpy())


@torch.no_grad()
def test(model,data,adj):
    #data.edge_label_index = data.pos_edge_label_index
    #data.edge_label = data.pos_edge_label
    # Now we concatenate the negative samples to the positive ones.
    #data.edge_label_index = torch.cat([data.edge_label_index, data.neg_edge_label_index], dim=-1)
    #data.edge_label = torch.cat([data.edge_label, data.edge_label.new_zeros(data.neg_edge_label_index.size(1))], dim=0)
    model.eval()
    z = model.encode(data.x, data.edge_index,adj)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy()),average_precision_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
