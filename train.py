#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                             
# PROGRAMMER: Bernd Schomburg                                                     
# DATE CREATED: 06/10/2018
# DATE REVISION 1: 06/10/2018
# DATE REVISION 2: 06/12/2018

import os, time, random, json, copy,  argparse
from collections import OrderedDict

import numpy as np

import torch
from torch import nn
from torch import optim
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms, models

from PIL import Image


def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() - data structure that stores the command line arguments object  
    """
    # Creates parse 
    parser = argparse.ArgumentParser()

    # Creates 8 command line arguments args.dir for data directory, architecture used for transfer learning, optional checkpoint,
    # number of epochs, learning rate, number of hidden units and CPU/GPU selection.
    
    parser.add_argument('--data_dir', type=str, 
                        help='Path to dataset (expects subdirectories \'train\' and \'valid\')')
    parser.add_argument('--arch', type=str, 
                        help='Network architecture', default = 'resnet18')
    parser.add_argument('--checkpoint', type=str, 
                        help='Save checkpoint to specified file')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=25)
    parser.add_argument('--lr', type=float, default = 0.01,
                        help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default = 1024,
                        help='Number of hidden units')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU if available')
    # returns parsed argument collection
    return parser.parse_args()


# Function that trains model
def train_model(model, criterion, optimizer, scheduler, device, dataloaders, dataset_sizes, num_epochs):
    ''' Trains model and prints out training and validation losses and accuracies per epoch.
        Parameters:
         model, criterion, optimizer, scheduler, device, dataloaders (directory of dataloaders with keys 'train' and 'valid'), 
         dataset_sizes (directory of dataset sizes with keys 'train' and 'valid'), num_epochs
        Returns:
         trained model.
    '''
    since = time.time()
    
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} loss: {:.4f} accuracy: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best validation accuracy: {:4f}'.format(best_acc))

    model.load_state_dict(best_model)
    return model

def main():
    in_arg = get_input_args()    

    # Set GPU if requested and available
    if in_arg.gpu and torch.cuda.is_available():
            print('Using GPU for prediction.')
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print('Using CPU for prediction.')
        if in_arg.gpu and not torch.cuda.is_available():
            print("Warning: GPU not available.")
       
    # Load and preprocess datasets (data augmentation and normalization for training, 
    # just normalization for validation and testing)
    if in_arg.data_dir and os.path.isdir(in_arg.data_dir):
        data_dir = in_arg.data_dir
    else:
        print("No dataset directory found at '{}'".format(in_arg.data_dir))    
        exit()
    if not os.path.isdir(data_dir+ '/train') or not os.path.isdir(data_dir + '/valid'):
        print("Dataset directory does not contain necessary subdirectories")    
        exit()
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    std_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            std_transform
            ]),
        'valid': std_transform,
        'test': std_transform
    }

    shuffle ={
        'train': True,
        'valid': False,
        'test': False
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'valid','test']}
    
    batch_size=in_arg.batch_size

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = batch_size,
                                             shuffle = shuffle[x])
                  for x in ['train', 'valid','test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid','test']}
    num_labels = len(image_datasets['train'].classes)
    print("Datasets successfully loaded")
    
    # Load a pre-trained model
    if in_arg.arch:
        arch = in_arg.arch

        if arch =='resnet18':
            model = models.resnet18(pretrained=True)
            num_ftrs = model.fc.in_features
        elif arch =='densenet121':
            model = models.densenet121(pretrained=True)
            num_ftrs = model.classifier.in_features    
        else:
            raise ValueError('Network architecture not supported', arch)
        print('Pretrained model {} loaded'.format(arch))
        
        # Freeze its parameters
        for param in model.parameters():
            param.requires_grad = False

        hidden_units = in_arg.hidden_units

        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(num_ftrs, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('d_out', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_units, num_labels)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        if arch == "resnet18":
            model.fc = classifier
        elif arch == "densenet121":
            model.classifier = classifier

        model.class_to_idx = image_datasets['train'].class_to_idx   
        model = model.to(device)
        
    criterion = nn.NLLLoss()

    optimizer = optim.SGD(classifier.parameters(), lr=in_arg.lr, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
     
    # Train model

    num_epochs = in_arg.epochs
        
    model = train_model(model, criterion, optimizer, exp_lr_scheduler, device = device, dataloaders = dataloaders, dataset_sizes = dataset_sizes, num_epochs= num_epochs)
      
    # Save model if checkpoint requested

    if in_arg.checkpoint:
        checkpoint = in_arg.checkpoint
        print ('Saving checkpoint to:', checkpoint) 

        checkpoint_dict = {
            'arch': arch,
            'classifier' : classifier,
            'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx,
            'hidden_units': hidden_units,
            'optimizer_dict': optimizer.state_dict(),
            'learning_rate': optimizer.state_dict()['param_groups'][0]['lr'], # = in_arg.lr * gamma**(num_epochs//step_size) 
            'epochs': num_epochs,
            'batch_size': batch_size
         }

        torch.save(checkpoint_dict, checkpoint)
    
if __name__ == '__main__':
    main()
