import os
import shutil

import ROOT
import numpy as np
import torch
import torch_geometric
from torch_geometric.datasets import FakeDataset
from torch_geometric.loader import DataLoader


def main():
    # Load data.
    np.random.seed(0)
    n_graphs = 300
    dataset = FakeDataset(num_graphs=n_graphs, avg_num_nodes=500, avg_degree=5, num_channels=32, num_classes=5)
    dataset = [graph for graph in dataset]
    global_attributes = torch.tensor(np.zeros(n_graphs), dtype=torch.float)
    for i in range(len(dataset)):
        dataset[i].put_tensor(global_attributes[i], attr_name='global_attr')
    train_dataset = dataset[:250]
    test_dataset = dataset[250:]
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Save test samples.
    shutil.rmtree('data')
    count = 0
    for data in test_loader:
        os.makedirs(f'data/batch_{count}')
        np.savetxt(f'data/batch_{count}/X.csv', data.x.numpy(), delimiter=',')
        np.savetxt(f'data/batch_{count}/edge_index.csv', data.edge_index.numpy(), delimiter=',')
        np.savetxt(f'data/batch_{count}/batch.csv', data.batch.numpy(), delimiter=',')
        count += 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch_model = Model().to(device)
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.001, weight_decay=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    def train():
        torch_model.train()

        for data in train_loader:  # Iterate in batches over the training dataset.
            data = data.to(device)
            out = torch_model(data.x, data.edge_index, data.batch)  # A single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

    @torch.no_grad()
    def test(loader, write=False):
        torch_model.eval()

        if write:
            with open('result.csv', 'w') as f:
                f.write('')

        correct = 0
        for data in loader:  # Iterate in batches over the test dataset.
            data = data.to(device)
            out = torch_model(data.x, data.edge_index, data.batch)

            if write:
                with open('result.csv', 'a') as f:
                    for i in out.cpu().numpy():
                        for j in i:
                            f.write(str(j) + ",")
                        f.write('\n')
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
    model = ROOT.TMVA.Experimental.SOFIE.RModel_TorchGNN(['X', 'edge_index', 'batch'], [[-1, 32], [2, -1], [-1]])
    model.AddModule(ROOT.TMVA.Experimental.SOFIE.RModule_GCNConv('X', 'edge_index', 32, 16), 'conv_1')
    model.AddModule(ROOT.TMVA.Experimental.SOFIE.RModule_ReLU('conv_1'), 'relu')
    model.AddModule(ROOT.TMVA.Experimental.SOFIE.RModule_GCNConv('relu', 'edge_index', 16, 16), 'conv_2')
    model.AddModule(ROOT.TMVA.Experimental.SOFIE.RModule_GlobalMeanPool('conv_2', 'batch'), 'pool')
    model.AddModule(ROOT.TMVA.Experimental.SOFIE.RModule_Linear('pool', 16, 5), 'linear')
    model.AddModule(ROOT.TMVA.Experimental.SOFIE.RModule_Softmax('linear'), 'softmax')
    model.ExtractParameters(torch_model)
    model.Save(os.getcwd(), 'Model', True)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.activation = torch.nn.ReLU()

        self.conv_1 = torch_geometric.nn.GCNConv(32, 16).jittable()
        self.conv_2 = torch_geometric.nn.GCNConv(16, 16).jittable()
        self.linear = torch.nn.Linear(16, 5)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, edge_index, batch):
        x = self.conv_1(x, edge_index)
        x = self.activation(x)
        x = self.conv_2(x, edge_index)
        x = torch_geometric.nn.global_mean_pool(x, batch)
        x = self.linear(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    main()
