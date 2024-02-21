import csv
import itertools

import torch
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='data/TUDataset', name='MUTAG')
# Sprawdzenie dostępności GPU
if torch.cuda.is_available():
    print("GPU dostępne")
    device = torch.device("cuda")
    print("Nazwa GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU niedostępne")
    device = torch.device("cpu")

# print()
# print(f'Dataset: {dataset}:')
# print('====================')
# print(f'Number of graphs: {len(dataset)}')
# print(f'Number of features: {dataset.num_features}')
# print(f'Number of classes: {dataset.num_classes}')
#
# data = dataset[0]  # Get the first graph object.
#
# print()
# print(data)
# print('=============================================================')
#
# # Gather some statistics about the first graph.
# print(f'Number of nodes: {data.num_nodes}')
# print(f'Number of edges: {data.num_edges}')
# print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
# print(f'Has isolated nodes: {data.has_isolated_nodes()}')
# print(f'Has self-loops: {data.has_self_loops()}')
# print(f'Is undirected: {data.is_undirected()}')
dataset = dataset.to(device)
dataset = dataset.shuffle()

train_dataset = dataset[:150]
test_dataset = dataset[150:]

# print(f'Number of training graphs: {len(train_dataset)}')
# print(f'Number of test graphs: {len(test_dataset)}')

from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# for step, data in enumerate(train_loader):
#     print(f'Step {step + 1}:')
#     print('=======')
#     print(f'Number of graphs in the current batch: {data.num_graphs}')
#     print(data)
#     print()

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(74873)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.lin(x)
        return x

#
# model = GCN(hidden_channels=64)
# model = model.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


# Dla grafowej
def train(_model, _optimizer):
    _model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = _model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        _optimizer.step()  # Update parameters based on gradients.
        _optimizer.zero_grad()  # Clear gradients.


def test(loader, _model):
    _model.eval()
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = _model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


if __name__ == "__main__":

    hidden_channels_list = [16, 32, 64]
    learning_rate_list = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
    weight_decay_list = [1e-4, 5e-4]
    num_epochs_list = [100, 200, 300, 500, 700]
    all_parameters_combination = list(
        itertools.product(hidden_channels_list, learning_rate_list, weight_decay_list, num_epochs_list))

    file_name = 'results_graph_level.csv'
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
            test_acc = test(test_loader,model)
            print(
                f'Index: {index}, Hidden channels: {hidden_channels}, Learning rate: {learning_rate}, Weight decay: {weight_decay}, '
                f'Number of epochs: {num_epochs}, Test Accuracy: {test_acc}')
            writer.writerow([index, hidden_channels, learning_rate, weight_decay, num_epochs,
                             test_acc])  # Zapisanie wyników do pliku

    # for epoch in range(1, 200):
    #     train()
    #     train_acc = test(train_loader)
    #     test_acc = test(test_loader)
    #     print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
