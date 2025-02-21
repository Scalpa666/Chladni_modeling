import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.model import AcDispNet
from models.dataset import DispDataset


def train(epoch, dataset, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Use device:', device)
    model.to(device)

    # Training setup
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    # Load the data for training
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Record the average loss per epoch
    loss_avg = []
    acc_rem = []
    maxdist_rem = []

    for t in range(epoch):
        total_loss = 0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            p = model(x)

            loss = loss_func(p, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if t % 100 == 0:
            accuracy, maxdist = acc(dataset, model)
            acc_rem.append([t] + accuracy)
            maxdist_rem.append([t] + maxdist)
            print('Epoch:', t, '| Loss:', total_loss / len(dataloader), '| Acc:', np.max(accuracy))

        if t % 10 == 0:
            loss_avg.append([t, total_loss / len(dataloader)])
        scheduler.step()

    # Plot the loss
    plt.figure()
    plt.plot([i[0] for i in loss_avg], [i[1] for i in loss_avg], marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.show()

    # Save the model
    torch.save(model.state_dict(), 'checkpoints/model.pth')

    # Save the data during training
    loss_df = pd.DataFrame(loss_avg, columns=['Epoch', 'Loss'])
    loss_df.to_csv('results/train/loss.csv', index=False)
    acc_df = pd.DataFrame(acc_rem)
    acc_df.to_csv('results/train/acc.csv', index=False)
    maxdist_df = pd.DataFrame(maxdist_rem)
    maxdist_df.to_csv('results/train/maxdist.csv', index=False)


def acc(dataset, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    predict_dist = np.zeros(12)
    max_predict_dist = np.zeros(12)
    min_predict_dist = np.ones(12) * 100
    predict_num = np.zeros(12)
    for x, y in dataset:
        x = x.to(device)
        y = y.to(device)
        p = model(x)
        dis = torch.sqrt(torch.sum((p - y) ** 2))
        data_class = int(x[0])
        predict_dist[data_class] += dis.item()
        predict_num[data_class] += 1
        if dis.item() > max_predict_dist[data_class]:
            max_predict_dist[data_class] = dis.item()
        if dis.item() < min_predict_dist[data_class]:
            min_predict_dist[data_class] = dis.item()
    predict_dist = predict_dist / predict_num
    predict_dist = predict_dist.tolist()
    max_predict_dist = max_predict_dist.tolist()
    print('Min: ', min_predict_dist)
    return predict_dist, max_predict_dist


if __name__ == "__main__":
    # Read data form the file
    train_data_path = './Data/train_data.csv'
    train_dataset = DispDataset(train_data_path)

    # print('Len of dataset:', len(dataset))
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])

    # Initialize the model
    model = AcDispNet(3, 2)
    # initial_loss = predict(dataset, model)

    # Train
    train(200, train_dataset, model)
