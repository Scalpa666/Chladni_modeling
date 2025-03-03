import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.model import *
from models.dataset import DispDataset


def train(total_epochs, dataset, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Use device:', device)
    model.to(device)

    # Training setup
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    # Load the data for training
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # Calculate losses before training
    model.eval()
    with torch.no_grad():
        initial_total_loss = 0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs  = model(inputs)
            initial_loss = loss_func(outputs , targets)
            initial_total_loss += initial_loss.item()
        initial_avg_loss = initial_total_loss / len(dataloader)
        print(f"Initial Loss before training: {initial_avg_loss:.4f}")

    model.train()

    # Record the average loss per epoch
    avg_loss_list = []
    metrics = RegressionMetrics(device)
    for epoch in range(total_epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            # Migrate data to the GPU
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward propagation
            outputs = model(inputs)
            loss = loss_func(outputs , targets)

            # Back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            metrics.update(outputs, targets)

        if epoch % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            mae, mse = metrics.compute()
            print(f"Epoch [{epoch + 1}/{total_epochs}] - Loss: {avg_loss} - MAE: {mae} - MSE: {mse:.4f}")
            avg_loss_list.append([epoch, avg_loss])

        # Periodically saves the model's training state
        if (epoch + 1) % 1000 == 0:
            checkpoint_path = f"checkpoints/model_epoch_{epoch + 1}.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, checkpoint_path)


        # scheduler.step()

    # Plot the loss
    plt.figure()
    plt.plot([i[0] for i in avg_loss_list], [i[1] for i in avg_loss_list], marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.show()

    # Save the model
    torch.save(model.state_dict(), 'checkpoints/model_final_1.pth')

    # Save the data during training
    loss_df = pd.DataFrame(avg_loss_list, columns=['Epoch', 'Loss'])
    loss_df.to_csv('results/train/loss.csv', index=False)


if __name__ == "__main__":
    # Read data form the file
    train_data_path = './Data/train_data.csv'
    train_dataset = DispDataset(train_data_path)

    # print('Len of dataset:', len(dataset))

    # Initialize the model
    model = AcDispNetL4(3, 2)
    # initial_loss = predict(dataset, model)

    # View initial weight
    print("Initial weight: ")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        print(model.state_dict()[param_tensor])

    # Train
    train(1000, train_dataset, model)
