import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.model import *
from models.dataset import DispDataset


def test(dataset, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Use device:', device)
    model.to(device)

    loss_func = nn.MSELoss()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    metrics = RegressionMetrics(device)
    total_loss = 0
    for inputs, targets in dataloader:
        # Migrate data to the GPU
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward propagation
        outputs = model(inputs)
        loss = loss_func(outputs, targets)

        total_loss += loss.item()
        metrics.update(outputs, targets)

    avg_loss = total_loss / len(dataloader)
    mae, mse = metrics.compute()
    print(f"Test Loss: {avg_loss} - MAE: {mae} - MSE: {mse:.4f}")

    # Plot the true value and predicted value to compare them
    inputs, targets = dataset[300:320]
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = model(inputs)

    inputs = inputs.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(inputs[:, -2], inputs[:, -1], targets[:, 1], c='r', label='Target')
    ax.scatter3D(inputs[:, -2], inputs[:, -1], outputs[:, 1], c='b', label='Predict')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    # Read data form the file
    test_data_path = './Data/test_data.csv'
    test_dataset = DispDataset(test_data_path)

    # Initialize the model
    model = AcDispNetL4(3, 2)

    # Test
    model.load_state_dict(torch.load('checkpoints/model_final_1.pth',  weights_only=True))
    model.eval()
    # valid_loss = test(train_dataset, model)
    test(test_dataset, model)