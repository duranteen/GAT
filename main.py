import numpy as np
import networkx as nx
import time
import torch
from torch.nn import functional as F
import dgl
from dgl.data import CoraGraphDataset

from gat import GAT

device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 0.02
weight_decay = 5e-4
num_epochs = 20
data = CoraGraphDataset()
graph = data[0]
features = graph.ndata['feat']
labels = graph.ndata['label']
train_mask = graph.ndata['train_mask']
val_mask = graph.ndata['val_mask']
test_mask = graph.ndata['test_mask']
num_feats = features.shape[1]
num_classes = data.num_labels
num_edges = data.graph.number_of_edges()
print("Edges {}, Classes {}, Train samples {}, Val samples {}, Test samples {}"
      .format(num_edges,
             num_classes,
             train_mask.sum().item(),
             val_mask.sum().item(),
             test_mask.sum().item())
)
graph = dgl.remove_self_loop(graph)
graph = dgl.add_self_loop(graph)
num_edges = graph.number_of_edges()
num_hidden = 256
num_layers = 3
num_heads = 10
num_out_heads = 5
feat_drop = 0.2
attn_drop = 0.2
negtive_slop = 0.2
residual = True
heads = ([num_heads] * (num_layers-1)) + [num_out_heads]
model = GAT(graph, num_layers, num_feats, num_hidden, num_classes, heads,
            F.relu(), feat_drop, attn_drop, negtive_slop, residual)
print(model)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fn = torch.nn.CrossEntropyLoss()

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)


def main():
    model.train()
    for epoch in range(num_epochs):
        logits = model(features)
        loss = loss_fn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.back_ward()
        optimizer.step()

        train_acc = accuracy(logits[train_mask], labels[train_mask])

        val_acc = evaluate(model, features, labels, val_mask)

        print("Epoch {:03d} Loss {:.4f} TrainAcc {:.4f} ValAcc {:.4f}"
              .format(epoch, loss.item(), train_acc, val_acc))

    acc = accuracy(model, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))

