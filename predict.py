#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                             
# PROGRAMMER: Bernd Schomburg                                                     
# DATE CREATED: 06/10/2018                                  
# DATE REVISION 1: 06/10/2018
# DATE REVISION 2: 06/12/2018

import os, time, json, copy, argparse
from collections import OrderedDict

import numpy as np

import torch
from torch import nn
import torchvision
from torchvision import transforms, models

from PIL import Image

# Functions defined below
def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. Returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Creates parse 
    parser = argparse.ArgumentParser()

    # Creates 5 command line arguments args.dir for paths to images file and checkpoint
    # to model to use for classification, to JSON file that maps class values to category names
    # and CPU/GPU selection.

    parser.add_argument('--file', type=str, 
                        help='path to image file')
    parser.add_argument('--checkpoint', type=str, 
                        help='Load model from saved checkpoint')
    parser.add_argument('--labels', type=str, 
                        help='Load JSON file that maps category labels to names')
    parser.add_argument('--topk', type=int, default =5,
                        help='top k predictions to be displayed')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU if available')

    # returns parsed argument collection
    return parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Tensor.
    '''
    std_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])
    
    pil_image = Image.open(image)
    tensor_image = std_transform(pil_image).float()
    
    return tensor_image

def load_checkpoint(filepath):
    ''' Loads a checkpoint and rebuilds the model for prediction.
    '''
    if torch.cuda.is_available():
        checkpoint = torch.load(filepath)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(filepath,
                                map_location=lambda storage,
                                loc: storage)      
    
    arch =  checkpoint['arch']
    if arch != "resnet18" and arch != "densenet121":
        raise ValueError('Network architecture not supported', arch)
    
    model=getattr(models, arch)(pretrained=True) 
    
    for param in model.parameters():
        param.requires_grad = False
 
    if arch == "resnet18":
        model.fc = checkpoint['classifier'] 
    else: # arch == "densenet121"
        model.classifier = checkpoint['classifier']
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def predict(image_path, model, device, topk=5):
    ''' Predicts the class (or classes) of an image using a trained deep learning model.
    '''
    with torch.no_grad():
        img_tensor = process_image(image_path)
        img_tensor = img_tensor.to(device)
        img_tensor.unsqueeze_(0) # resize the tensor (add dimension for batch)

        model = model.to(device)
        model.eval()   # Set model to evaluate mode
  
    # apply data to model
        output = model(img_tensor).topk(topk)
        probs = torch.exp(output[0].to("cpu")).numpy().tolist()[0]
        classes = (output[1].to("cpu")).numpy().tolist()[0]
    
    return probs, classes

def main():
        
    in_arg = get_input_args()
    
    # Sets GPU if requested and available    
    if in_arg.gpu and torch.cuda.is_available():
            print('Using GPU for prediction.')
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print('Using CPU for prediction.')
        if in_arg.gpu and not torch.cuda.is_available():
            print("Warning: GPU not available.")
            
    # Load a mapping from category label to category name
    if in_arg.labels and os.path.isfile(in_arg.labels):
        with open(in_arg.labels, 'r') as f:
            cat_to_name = json.load(f)
    # Load model
    if in_arg.checkpoint and os.path.isfile(in_arg.checkpoint):
            print("Loading checkpoint '{}'".format(in_arg.checkpoint))
            model = load_checkpoint(in_arg.checkpoint)
    else:
        print("No checkpoint found at '{}' ".format(in_arg.checkpoint))
        exit()

    # Load image and predict and print topk predictions
    if in_arg.file and os.path.isfile(in_arg.file):
        image = in_arg.file
        print("Loading image file '{}'".format(in_arg.file))
    else:
        print("No file found at '{}'".format(in_arg.file))    
        exit()
        
    probs, classes = predict(image, model, device = device, topk=in_arg.topk)   
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    names =[] 
    
    for i in range(len(classes)):
        if in_arg.labels:
            names.append(cat_to_name[idx_to_class[classes[i]]])  
        else:
            names.append(idx_to_class[classes[i]])
            
    print('The {} most probable predictions (in descending order) are:'.format(in_arg.topk))
    
    print('#'+2*' '+  'name/class' + 15*' '+'probability')
    for i in range(len(classes)):
        print('{:d}  {:25s}{:.5f}'.format(
                i+1, names[i], probs[i]))

        
if __name__ == '__main__':
    main()