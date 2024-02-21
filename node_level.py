# Install required packages.
import itertools
import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv
import csv

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

import torch
from torch.nn import Linear

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch

# Sprawdzenie dostępności GPU
if torch.cuda.is_available():
    print("GPU dostępne")
    device = torch.device("cuda")
    print("Nazwa GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU niedostępne")
    device = torch.device("cpu")


def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


print()
# print(f'Dataset: {dataset}:')
# print('======================')
# print(f'Number of graphs: {len(dataset)}')
# print(f'Number of features: {dataset.num_features}')
# print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.


# print()
# print(data)
# print('===========================================================================================================')

# Gather some statistics about the graph.
# print(f'Number of nodes: {data.num_nodes}')
# print(f'Number of edges: {data.num_edges}')
# print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
# print(f'Number of training nodes: {data.train_mask.sum()}')
# print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
# print(f'Has isolated nodes: {data.has_isolated_nodes()}')
# print(f'Has self-loops: {data.has_self_loops()}')
# print(f'Is undirected: {data.is_undirected()}')

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(dataset.num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


#
# model = MLP(hidden_channels=16)
# print(model)
# criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.
#
#
# def train():
#     model.train()
#     optimizer.zero_grad()  # Clear gradients.
#     out = model(data.x)  # Perform a single forward pass.
#     loss = criterion(out[data.train_mask],
#                      data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
#     loss.backward()  # Derive gradients.
#     optimizer.step()  # Update parameters based on gradients.
#     return loss
#
#
# def test():
#     model.eval()
#     out = model(data.x)
#     pred = out.argmax(dim=1)  # Use the class with highest probability.
#     test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
#     test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
#     return test_acc
#
#
# for epoch in range(1, 201):
#     loss = train()
#     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
#
#
# test_acc = test()
# print(f'Test Accuracy: {test_acc:.4f}')


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x


def train(_model, _optimizer):
    _model.train()
    _optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    _optimizer.step()  # Update parameters based on gradients.
    return loss


def test(_model):
    _model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc


if __name__ == "__main__":
    # hidden_channels = 32
    # learning_rate = 0.01
    # weight_decay = 5e-4
    # num_epochs = 300
    # model = GCN(hidden_channels=hidden_channels)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    hidden_channels_list = [16, 32, 64]
    learning_rate_list = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
    weight_decay_list = [1e-4, 5e-4, 1e-2, 5e-2]
    num_epochs_list = [100, 200, 300, 500, 700]
    all_parameters_combination = list(
        itertools.product(hidden_channels_list, learning_rate_list, weight_decay_list, num_epochs_list))

    file_name = 'results_node_level.csv'
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["index", "hidden_channels", "learning_rate", "weight_decay", "num_epochs", "test_accuracy"])
        for index, parameters in enumerate(all_parameters_combination):
            hidden_channels, learning_rate, weight_decay, num_epochs = parameters  # Ustawienie parametrow
            model = GCN(hidden_channels=hidden_channels)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            for epoch in range(num_epochs):  # Trenowanie
                loss = train(_model=model, _optimizer=optimizer)

            test_acc = test(_model=model)  # Testowanie

            print(
                f'Index: {index}, Hidden channels: {hidden_channels}, Learning rate: {learning_rate}, Weight decay: {weight_decay}, '
                f'Number of epochs: {num_epochs}, Test Accuracy: {test_acc}')
            writer.writerow([index, hidden_channels, learning_rate, weight_decay, num_epochs,
                             test_acc])  # Zapisanie wyników do pliku

    # model.eval()
#   out = model(data.x, data.edge_index)
#   visualize(out, color=data.y)
