from tokenize import PlainToken
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train(model, trainloader, optimizer, device, valloader = None, num_epochs = 100, val = False, plot = True, printout = True):
    """Trains a model and optionally performs validation as well"""
    
    model.to(device)
    
    # loss
    loss_fn = nn.CrossEntropyLoss()
    
    ### Training loop ###
    TRAIN_LOSS, TRAIN_ACC = [], []
    VAL_LOSS, VAL_ACC = [], []

    model.train()

    for epoch in range(num_epochs):
        train_loss = 0
        train_correct = 0
        train_total = 0
        for X, y, idx in trainloader:
            # Get data to device if possible
            X = X.to(device)
            y = y.to(device)
            
            # forward
            output = model(X)
            loss = loss_fn(output, y)
            train_loss += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()
            
            preds = torch.argmax(F.softmax(output, dim = 1),dim=1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
        
        train_loss_epoch = train_loss/len(trainloader)
        train_acc_epoch = (train_correct/train_total)*100
        
        TRAIN_LOSS.append(train_loss_epoch)
        TRAIN_ACC.append(train_acc_epoch)
        
        val_total = 0
        val_loss = 0
        val_correct = 0
        
        if val:
            with torch.no_grad():
                for X_val, y_val, idx in valloader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    output = model(X_val)
                    loss = loss_fn(output, y_val)
                    val_loss += loss.item()  # sum up batch loss
    
                    preds = torch.argmax(F.softmax(output, dim = 1),dim=1)
                    val_correct += (preds == y_val).sum().item()
                    val_total += y_val.size(0)

            val_loss_epoch = train_loss/len(valloader)
            val_acc_epoch = (val_correct/val_total)*100
            
            VAL_LOSS.append(val_loss_epoch)
            VAL_ACC.append(val_acc_epoch)
                
        if printout:
            print(f'Epoch: {epoch:3d} | Train Loss: {train_loss_epoch:.2f} | Train Acc: {train_acc_epoch:.1f}%')

    if plot:  
        fig, (ax1, ax2) = plt.subplots(1,2, sharex = 'col', figsize=(8,4))
        ax1.plot(TRAIN_LOSS, label = 'train')
        ax1.plot(VAL_LOSS, label = 'val')
        ax1.legend()
        ax1.set_title('LOSS')

        ax2.plot(TRAIN_ACC, label = 'train')
        ax2.plot(VAL_ACC, label = 'val')
        ax2.legend()
        ax2.set_title('ACCURACY')
        plt.show()
    
    return model
    
def test(model, testloader, device, display=True):
    model.eval()
    
    test_loss = 0
    n_correct = 0
    total = 0
    
    loss_fn = nn.CrossEntropyLoss()
    
    TEST_ACC = []
    with torch.no_grad():
        for X, y, idx in testloader:
            
            X, y= X.to(device), y.to(device)
            out = model(X)
            
            test_loss += loss_fn(out, y).item()  # sum up batch loss
            preds = torch.argmax(F.softmax(out, dim = 1),dim=1)
            n_correct += (preds == y).sum().item()
            total += y.size(0)

    loss = test_loss/len(testloader)
    acc = (n_correct/total)
    
    if display:
        print(f'Accuracy on the test set: {acc:.1%} %')
    
    return acc