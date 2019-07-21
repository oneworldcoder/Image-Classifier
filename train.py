import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import time
import copy
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import json
import torchvision.models as models
from PIL import Image
import argparse

# Arguement Parser

ague_p = argparse.ArgumentParser(description='Train.py')

# Command Line ardguments

ague_p.add_argument('--data_dir', type=str, action="store", default="flowers/")
ague_p.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint1.pth")
#if --arch is other than 'vgg19' then choose your input units .
ague_p.add_argument('--arch', dest='arch', action="store", default="vgg19", type = str)
ague_p.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ague_p.add_argument('--input_units', type=int, dest="input_units", action="store", default= 25088)
ague_p.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096)
ague_p.add_argument('--output_units', type=int, dest="output_units", action="store", default=102)
ague_p.add_argument('--epochs', dest="epochs", action="store", type=int, default=4)
ague_p.add_argument('--dropout', dest = "dropout", action = "store", default = 0.3)
ague_p.add_argument('--gpu', help= "Option to use GPU.", type= str)

p_ague = ague_p.parse_args()

data_dir = p_ague.data_dir
train_dir = data_dir + 'train/'
valid_dir = data_dir + 'valid/'
test_dir = data_dir + 'test/'
save_dir = p_ague.save_dir
arch = p_ague.arch
learning_rate = p_ague.learning_rate
input_units = p_ague.input_units 
hidden_units = p_ague.hidden_units
dropout = p_ague.dropout
output_units = p_ague.output_units
epochs = p_ague.epochs
gpu = p_ague.gpu

data_transforms = {'train_dir': transforms.Compose([transforms.RandomRotation(30), transforms.RandomHorizontalFlip(),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225])]) ,
                   'valid_dir': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                       [0.229, 0.224, 0.225])]) ,
                   'test_dir':transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                                  transforms.ToTensor(), 
                                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])]) }

image_datasets = {'train_dir': datasets.ImageFolder(train_dir, transform = data_transforms['train_dir']),
                 'valid_dir': datasets.ImageFolder(valid_dir, transform = data_transforms['valid_dir']),
                 'test_dir': datasets.ImageFolder(test_dir, transform = data_transforms['test_dir'])}

dataloaders = {'train_dir': torch.utils.data.DataLoader(image_datasets['train_dir'], batch_size= 64, shuffle = True),
              'valid_dir': torch.utils.data.DataLoader(image_datasets['valid_dir'], batch_size= 64),
              'test_dir': torch.utils.data.DataLoader(image_datasets['test_dir'], batch_size= 64)}

if p_ague.gpu == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

if p_ague.arch == 'vgg19':
    model = models.vgg19(pretrained= True)
if p_ague.arch == 'alexnet':#input_units should be 9216
    model = models.alexnet(pretrained = True)


for param in model.parameters():
    param.requires_grad = False
    

model.classifier = nn.Sequential(nn.Linear(p_ague.input_units, p_ague.hidden_units),
                                 nn.Dropout(p_ague.dropout),
                                 nn.ReLU(),
                                 nn.Linear(p_ague.hidden_units, p_ague.output_units),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr = p_ague.learning_rate)

model.to(device)


epochs = p_ague.epochs
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in dataloaders['train_dir']:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in dataloaders['valid_dir']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print("Epoch ..{}/{} ".format((epoch+1), (epochs)))
            print("Train loss: {:.3f}".format(running_loss/print_every))
            print("Validation loss: {:.3f} ".format(test_loss/len(dataloaders['valid_dir'])))
            print("Validation accuracy: {:.3f}%".format(100*accuracy/len(dataloaders['valid_dir'])))
            running_loss = 0
            model.train()

#save checkpoint
model.to('cpu')

model.class_to_idx = image_datasets['train_dir'].class_to_idx


checkpoint = {'arch' : p_ague.arch,
              'state_dict': model.state_dict(),
              'class_to_idx' : model.class_to_idx}
print(p_ague.save_dir)        
torch.save(checkpoint, p_ague.save_dir)
