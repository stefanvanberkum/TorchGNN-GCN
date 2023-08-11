import numpy as np
import ROOT
import torch
import torch_geometric
from torch.utils.data import DataLoader
from torch_geometric.datasets import FakeDataset


def main():
    # Load data.
    n_graphs = 300
    dataset = FakeDataset(num_graphs=n_graphs)
    dataset = [graph for graph in dataset]
    global_attributes = torch.tensor(np.zeros(n_graphs), dtype=torch.float)
    for i in range(len(dataset)):
        dataset[i].put_tensor(global_attributes[i], attr_name='global_attr')
    train_dataset = dataset[:250]
    test_dataset = dataset[250:]
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Save test samples.
    np.savetxt('data/X.csv', X_test, delimiter=',')
    np.savetxt('data/y.csv', y_test, delimiter=',')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    def train():
        model.train()

        for data in train_loader:  # Iterate in batches over the training dataset.
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.global_attr, data.batch)  # A single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

    @torch.no_grad()
    def test(loader):
        model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.global_attr, data.batch)
            pred = out.argmax(dim=1)  # Use the class with the highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.

    for epoch in range(1, 50):
        train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    _ = test(test_loader, True)

    # Save state dictionary.
    torch.save(torch_model.state_dict(), 'model_dict.pt')

    # Save model.
    model_script = torch.jit.script(torch_model)
    torch.jit.save(model_script, 'model_script.pt')

    # Generate ROOT model.
    model = ROOT.TMVA.Experimental.SOFIE.RModel_TorchGNN(['X'], [[-1, 3072]])
    model.addModule(ROOT.TMVA.Experimental.SOFIE.RModule_Linear('X', 3072, 200), 'linear_1')
    model.addModule(ROOT.TMVA.Experimental.SOFIE.RModule_ReLU('linear_1'), 'relu_1')
    model.addModule(ROOT.TMVA.Experimental.SOFIE.RModule_Linear('relu_1', 200, 200), 'linear_2')
    model.addModule(ROOT.TMVA.Experimental.SOFIE.RModule_ReLU('linear_2'), 'relu_2')
    model.addModule(ROOT.TMVA.Experimental.SOFIE.RModule_Linear('relu_2', 200, 10), 'linear_3')
    model.addModule(ROOT.TMVA.Experimental.SOFIE.RModule_Softmax('linear_3'), 'softmax')
    model.extractParameters(torch_model)
    model.save("/home/stefan/TorchGNN", "Model", True)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.activation = torch.nn.ReLU()

        self.conv1 = torch_geometric.nn.GCNConv(64, 16)
        self.conv2 = torch_geometric.nn.GCNConv(16, 16)
        self.linear = torch.nn.Linear(16, 10)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, edge_index, batch_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        x = torch_geometric.nn.global_mean_pool(x, batch_index)
        x = self.linear(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    main()
