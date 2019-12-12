#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 16:31:06 2019

@author: masonperry
"""
### Script to start working on a multilayer preceptron to determine volcanoes on venus
### This script is to extract data for each image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset



## Load Training and test data
loc = '~/Documents/School/Machine_Learning/project/volcanoesvenus/'

x_train = torch.from_numpy(np.array(pd.read_csv('train_images.csv',header=None))/255.)
y_train = torch.from_numpy(np.array(pd.read_csv('train_labels.csv'))[:,0]) #to get labels
x_test = torch.from_numpy(np.array(pd.read_csv('test_images.csv',header=None))/255.)
y_test = torch.from_numpy(np.array(pd.read_csv('test_labels.csv'))[:,0]) #to get labels

x_train = x_train.to(torch.float32)
x_test = x_test.to(torch.float32)
y_train = y_train.to(torch.long)
y_test = y_test.to(torch.long)

## Set Device
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

### Initialize Neural Network
n_classes = 2
n = x_train.shape[1]
batch_size = 100 #1 for stochastic gradient descent
learning_rate = 1e-4
epochs = 150 #if run for too long, it might be memorizing a bit
##Make the Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1 = nn.Linear(n,16)
        self.l2 = nn.Linear(16,8)
        self.l3 = nn.Linear(8,n_classes)
    def forward(self,x):
        x = torch.cos(self.l1(x))
        ## ReLu is best found so far, getting up to 93% accuracy in testset
        ## cosh:85 sin&cos:92ish closest to relu tan:60ish sigmoid and tanh: 85ish
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return x
        


###Run from here unless you modified the model
training_data = TensorDataset(x_train,y_train)
test_data = TensorDataset(x_test,y_test)
train_loader = torch.utils.data.DataLoader(dataset=training_data,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False)

model = Net()
model.to(device)
criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

## For all loss values
L_all=[]

# Loop over the data
for epoch in range(epochs):
    # Loop over each subset of data
    total = 0 
    correct = 0
    for d,t in train_loader:
        # Zero out the optimizer's gradient buffer
        optimizer.zero_grad()
        
        # Make a prediction based on the model
        outputs = model(d)
        
        # Compute the loss
        loss = criterion(outputs,t)

        # Use backpropagation to compute the derivative of the loss with respect to the parameters
        loss.backward()
        
        # Use the derivative information to update the parameters
        optimizer.step()
        _, predicted = torch.max(outputs.data,1)
        total += t.size(0)
        correct += (predicted==t).sum()
    print('epoch:',epoch,'loss:',loss.item(),'Training Accuracy is:',(100.*correct/total).item())
    # After each epoch, compute the test set accuracy
    total=0
    correct=0
    # Loop over all the test examples and accumulate the number of correct results in each batch
    for d,t in test_loader:
        outputs = model(d)
        _, predicted = torch.max(outputs.data,1)
        total += t.size(0)
        correct += (predicted==t).sum()
        
    # Print the epoch, the training loss, and the test set accuracy.
    L_all.append(loss.item())
    print('Test Accuracy is:',(100.*correct/total).item())
plt.plot(L_all)
plt.show()
print('Accuracy is:',(100.*correct/total).item(),'%')
plt.plot(L_all)
plt.show()


outputs_test = model(x_test)
_, predicted = torch.max(outputs_test.data,1)


## Generate indices of each, could clean up code a bit
# Correctly identified as having a volcano
correct_volc=[]
for i in range(len(predicted)):
    if int(predicted[i]) == 1:
        if int(y_test[i]) == 1:
            correct_volc.append(i)
# Correctly identified as not having a volcano
correct_no_volc=[]
for i in range(len(predicted)):
    if int(predicted[i]) == 0:
        if int(y_test[i]) == 0:
            correct_no_volc.append(i)
# Incorrectly identified no volcano, actually does
incorrect_act_volc=[]
for i in range(len(predicted)):
    if int(predicted[i]) == 0:
        if int(y_test[i]) == 1:
            incorrect_act_volc.append(i)
# Incorrectly identified having a volcano, actually doesnt
incorrect_act_no_volc=[]
for i in range(len(predicted)):
    if int(predicted[i]) == 1:
        if int(y_test[i]) == 0:
            incorrect_act_no_volc.append(i)

## Plot up some random ones        
fig,axs = plt.subplots(2,2,figsize=(8,8))
temp = np.random.choice(correct_volc)
axs[0,0].imshow(x_test[temp].reshape(110,110),vmin=0,vmax=1)
axs[0,0].set_title('Correctly identified with volcano')
temp = np.random.choice(correct_no_volc)
axs[0,1].imshow(x_test[temp].reshape(110,110),vmin=0,vmax=1)
axs[0,1].set_title('Correctly identified without volcano')
temp = np.random.choice(incorrect_act_volc)
axs[1,0].imshow(x_test[temp].reshape(110,110),vmin=0,vmax=1)
axs[1,0].set_title('Incorrectly identified, has volcano')
temp = np.random.choice(incorrect_act_volc)
axs[1,1].imshow(x_test[temp].reshape(110,110),vmin=0,vmax=1)
axs[1,1].set_title('Incorrectly identified, has no volcano')
plt.show()

##Plot up the weights from 1st hidden layer
params = [p.cpu().detach().numpy() for p in model.l1.parameters()]
W = params[0]
for i in range(W.shape[0]):
    plt.imshow(W[i,:].reshape([110,110]))
    plt.title('Example Weight Image')
    plt.show()
    






