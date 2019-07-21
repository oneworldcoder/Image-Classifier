import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
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


ague_p = argparse.ArgumentParser(description='predict-file || standard command: python predict.py flowers/test/102/image_08030.jpg')

ague_p.add_argument('--arch',  action="store", default="vgg19", type = str)
ague_p.add_argument('--learning_rate', type=int, action="store", default=0.001)
ague_p.add_argument('--input_units', type=int, action="store", default= 25088)
ague_p.add_argument('--hidden_units', type=int,  action="store", default=4096)
ague_p.add_argument('--output_units', type=int,  action="store", default=102)
ague_p.add_argument('--epochs', dest="epochs", action="store", type=int, default=4)
ague_p.add_argument('--dropout', dest = "dropout", action = "store", default = 0.3)
ague_p.add_argument('--image_path', action="store", type = str, default= 'flowers/test/102/image_08030.jpg')
ague_p.add_argument('--checkpoint', default= 'checkpoint1.pth', action="store",type = str)
ague_p.add_argument('--top_k', default=5,  action="store", type=int)
ague_p.add_argument('--category_names', type= str, action="store", default='cat_to_name.json')
ague_p.add_argument('--gpu', help= "Option to use GPU." , type= str)


p_ague = ague_p.parse_args()
arch = p_ague.arch
image_path = p_ague.image_path
number_of_outputs = p_ague.top_k
power = p_ague.gpu
top_k = p_ague.top_k
checkpoint = p_ague.checkpoint
input_units = p_ague.input_units 
hidden_units = p_ague.hidden_units
dropout = p_ague.dropout
output_units = p_ague.output_units
epochs = p_ague.epochs
category_names = p_ague.category_names
gpu = p_ague.gpu

if p_ague.gpu == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'

if p_ague.category_names:
    with open(p_ague.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass

def load_checkpoint(filepath):
    print(filepath)
    checkpoint = torch.load(filepath)
    print(checkpoint.keys())
    if checkpoint['arch'] == 'vgg19':
        model = models.vgg19(pretrained = True)
    elif checkpoint['arch'] == 'alexnet':    
        model = models.alexnet(pretrained = True)
    else:
        print('Model Not recognised')

    for param in model.parameters():
        param.requires_grad = False
        
    model.class_to_idx = checkpoint['class_to_idx']
    
    model.classifier = nn.Sequential(nn.Linear(p_ague.input_units, p_ague.hidden_units),
                                 nn.Dropout(p_ague.dropout),
                                 nn.ReLU(),
                                 nn.Linear(p_ague.hidden_units,  p_ague.output_units),
                                 nn.LogSoftmax(dim=1))
    model.load_state_dict(checkpoint['state_dict'])
    return model

model = load_checkpoint(p_ague.checkpoint)

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    '''
    test_image = Image.open(p_ague.image_path)
    if test_image.size[0] > test_image.size[1]:
        test_image.thumbnail((22000, 256))
    else: 
        test_image.thumbnail((256, 22000))
    left_test_image = (test_image.width - 224) / 2
    below_test_image = (test_image.height - 224) / 2    
    right_test_image = left_test_image + 224 
    above_test_image = below_test_image + 224
    test_image = test_image.crop((left_test_image, below_test_image,
                                right_test_image, above_test_image))
    test_image = np.array(test_image) / 255
    mean_test_image = np.array([0.485, 0.456, 0.406])
    sd_test_image = np.array([0.229, 0.224, 0.225])
    test_image = (test_image - mean_test_image) / (sd_test_image)
    test_image = test_image.transpose((2, 0, 1))
    return test_image

check_image = process_image(p_ague.image_path)


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = np.array(image)
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    return ax

imshow(check_image)

#class predicton

def predict(image_path, model, top_k, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    test_image = process_image(p_ague.image_path)
    
    if device == 'cuda':
        numpy_to_tensor = torch.from_numpy(test_image).type(torch.cuda.FloatTensor)
    else:
        numpy_to_tensor = torch.from_numpy(test_image).type(torch.FloatTensor)
    
    batch_input = numpy_to_tensor.unsqueeze(0)
    
    model.to(device)
    numpy_to_tensor.to(device)
    
    ps = torch.exp(model.forward(batch_input))
    
    top_p, top_class = ps.topk(top_k)
    
    top_p = top_p.cpu()
    top_class = top_class.cpu()
    
    top_p = top_p.detach().numpy().tolist()[0]
    
    top_class = top_class.detach().numpy().tolist()[0]
    
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    
    likely_labels = [idx_to_class[c] for c in top_class]
    return top_p, likely_labels

top_p, likely_labels = predict(image_path, model, top_k, device)


class_labels = [cat_to_name[i] for i in likely_labels]

if p_ague.top_k:
    t_k = p_ague.top_k
else:
    t_k = 1

for a in range(t_k):
     print("Number: {}/{}.. ".format(a+1, t_k))
     print("Class Labels: {}.. ".format(class_labels[a]))
     print("Probability: {:.2f}..% ".format(top_p[a]*100))