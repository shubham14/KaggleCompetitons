# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 12:13:32 2019

@author: Shubham
"""

from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('C:\\Users\\Shubham\\Desktop\\Fraud_Detection\\helpers')
from config import cfg
from data_process import *
import time

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(cfg.image_channels, 32, 3)
        self.cnn2 = nn.Conv2d(32, 64, 3)
        self.linear1 = nn.Linear(589824, 1)
    
    def forward(self, inp):
        out = self.cnn1(inp)
        out = self.cnn2(out)
        out = Flatten()(out)
        out = F.sigmoid(self.linear1(out))
        return out
        

def train(trainloader, net, criterion, optimizer, device, scheduler):
    net = net.double()
    for epoch in range(100):  # loop over the dataset multiple times
        scheduler.step()
        start = time.time()
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            images = data['image'].to(device)
            labels = data['isFraud'].to(device)
            # TODO: zero the parameter gradients
            # TODO: forward pass
            # TODO: backward pass
            # TODO: optimize the network
            optimizer.zero_grad()
            out = net(images)
            loss = criterion(out, labels.double())
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                end = time.time()
                print('[epoch %d, iter %5d] loss: %.3f eplased time %.3f' %
                      (epoch + 1, i + 1, running_loss / 100, end-start))
                start = time.time()
                running_loss = 0.0
    print('Finished Training')
    
    print("Saving model information")
    torch.save(model.state_dict(), "cnn_model.pth")

def test(testloader, net, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data['image'], data['isFraud']
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

if __name__ == "__main__":
    
    if torch.cuda.is_available() and cfg.device == 'cuda':
        print("Running on GPU!!, Yay")
        
    modes = ['train', 'test']
    pandas_df = []
    for mode in modes:
        x = combine_identity_transaction(mode=mode)
        pandas_df.append(x)
    X_train, y_train, X_test = processDataFrame(pandas_df[0], 
                                                pandas_df[1])
    
    print('here')
    # dataset instantiating
    train_loader = DataLoader(TransactionDataset(X_train, y_train, mode='S'),
                              batch_size=cfg.batch_size, shuffle=True)
    
    # instantiating the model
    cnn = CNN().to(cfg.device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(cnn.parameters(), lr=0.1, momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=[25, 50, 75], gamma=0.08)
    train(train_loader, cnn, criterion, optimizer, cfg.device, scheduler)
#    test(testloader, cnn, cfg.device)