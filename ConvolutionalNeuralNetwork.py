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

        return x

#training function to train model
def training(model, loader, optimiser, lossfunc):
    model.train() # train mode

    totalloss = 0
    correct = 0
    total = 0


    for images,labels in loader: #amount of images is the batchsize defined in main

        # forward pass, batch flows through conv layers and fc layers
        outputs = model(images)

        #find loss, uses loss func defined in main
        loss = lossfunc(outputs, labels)

        #clear grads from prev batch of images
        optimiser.zero_grad()

        #backpropagation
        loss.backward()

        #update weights using optimiser in main
        optimiser.step()

        #total loss
        totalloss += loss.item()

        #accuracy calculation
        _, predicted = torch.max(outputs, 1) #the underscore here means that that variable does not mean anything,
        #torch.max returns the index of the max into predicted, which conveniently is the same as the predicted class
        correct += torch.sum(predicted == labels.data) #if its correct add 1 to correct, since true == 1
        total += labels.size(0) # batch size

    meanloss = totalloss / len(loader)
    accuracy = 100*(correct / total)

    return meanloss, accuracy

def evaluating(model, loader, lossfunc):
    model.eval() #eval mode
    totalloss = 0
    correct = 0
    total = 0

    #testing/evaluating doesnt require gradients since those are calculated during training, hence why no_grad
    #backpropagation is not required either.

    with torch.no_grad():
        for images,labels in loader:
            outputs = model(images)
            loss = lossfunc(outputs, labels)
            totalloss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += torch.sum(predicted == labels.data)
            total += labels.size(0)



    meanloss = totalloss / len(loader)
    accuracy = 100*(correct / total)
    return meanloss, accuracy


def main():

    #seed for reproducibility
    torch.manual_seed(1)

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

    #initialising model, loss function, and optimiser
    model = ConvolutionalNeuralNetwork()
    lossfunc = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01)

    #running the training and evaluation loop

    epochs = 10
    for epoch in range(epochs):
        trainloss, trainacc = training(model, trainLoader, optimiser, lossfunc)
        testloss, testacc = evaluating(model, testLoader, lossfunc)

        print("Epoch ",epoch+1," / ",epochs)
        print("Train Loss: ",trainloss," Train Accuracy: ",trainacc)
        print("Test Loss: ",testloss," Test Accuracy: ",testacc)
        print("")

if __name__ == "__main__":
    main()
