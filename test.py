import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.model import *
from models.dataset import DispDataset


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


def test(dataset, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Use device:', device)
    model.to(device)

    loss_func = nn.MSELoss()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # accuracy, max_dist = acc(dataset, model)
    # class_std = np.zeros(12)
    # predict_num = np.zeros(12)
    total_loss = 0
    #
    # for x, y in dataset:
    #     x = x.to(device)
    #     y = y.to(device)
    #     p = model(x)
    #
    #     loss = loss_func(p, y)
    #     total_loss += loss.item()
    #
    #     dis = torch.sqrt(torch.sum((p - y) ** 2))
    #
    #     data_class = int(x[0])
    #     predict_num[data_class] += 1
    #     class_std[data_class] += (accuracy[data_class] - dis.item()) ** 2
    # class_std = class_std / predict_num
    #
    # print('Mse Loss:', total_loss / len(dataloader))
    # print('Accuracy:', accuracy)
    # print('Std:', class_std)
    # print('Max: ', max_dist)

    # Plot the true value and predicted value to compare them
    x, y = dataset[300:320]
    x = x.to(device)
    y = y.to(device)
    p = model(x)

    x = x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    p = p.cpu().detach().numpy()

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x[:, -2], x[:, -1], y[:, 1], c='r', label='Input')
    ax.scatter3D(x[:, -2], x[:, -1], p[:, 1], c='b', label='Predict')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.legend()
    plt.show()

    return total_loss / len(dataloader)


# Read data form the file
test_data_path = './Data/test_data.csv'
test_dataset = DispDataset(test_data_path)

# Initialize the model
model = AcDispNetL8(3, 2)

# Test
model.load_state_dict(torch.load('checkpoints/model_final.pth'))
model.eval()
# valid_loss = test(train_dataset, model)
test_loss = test(test_dataset, model)