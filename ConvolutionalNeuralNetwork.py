import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        #conv layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1 )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        #pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #fully connected layers
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        #conv1 layer
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        #conv2 layer
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        #flatten
        x = x.view(-1, 64*7*7)
        #fully connected layers
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        y = self.final(x)
        return y

    def main(self):

        #MNIST loading and transforming here, mean and std provided with MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        #get test and training data from torchvision
        trainDataset = datasets.MNIST(
            root='./data',
            train=True,
            transform=transform,
            download=True)

        testDataset = datasets.MNIST(
            root='./data',
            train=False,
            transform=transform,
            download=True)

        #check if data is correctly loaded
        print(len(trainDataset))
        print(len(testDataset))

        #loading the data
        trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True)
        testLoader = DataLoader(testDataset, batch_size=64, shuffle=False)

        #model
        model = ConvolutionalNeuralNetwork()
        loss = nn.CrossEntropyLoss()
        