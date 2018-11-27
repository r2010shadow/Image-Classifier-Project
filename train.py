# Imports here
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from collections import OrderedDict
import torchvision
from PIL import Image
import argparse,copy,os,random,time
from public import save_checkpoint, load_checkpoint


def train_model(model, criterion, optimizer, scheduler, epochs, gpu):
    # TODO: Build and train your network
    use_gpu = torch.cuda.is_available()
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc       = 0.0    
    
    if gpu and use_gpu:
        model.cuda()
        print("Use GPU. Training ...")
    else:
        model.cpu()
        print("Use CPU. Training ...")
        

    for e in range(epochs):   
        print('Epoch {}/{}'.format(e+1, epochs))
        for param in ['train', 'valid']:
            if param == 'train':
                scheduler.step()
                model.train(True) 
            else:
                model.train(False) 

            running_loss     = 0.0
            running_corrects = 0

            for data in dataloaders[param]:
                inputs, labels = data
                
                if gpu and use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)
                
                optimizer.zero_grad()

                # forward
                outputs  = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss     = criterion(outputs, labels)

                if param == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss     += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)                


            epoch_loss = running_loss / dataset_sizes[param]
            epoch_acc  = running_corrects.double() / dataset_sizes[param]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(param, epoch_loss, epoch_acc))

            if param == 'valid' and epoch_acc > best_acc:
                best_acc       = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model       


def todo_args():
    # TODO: Help to run the project
    parser = argparse.ArgumentParser(description="Image-Classifier Project")
    parser.add_argument('data_dir' ,  default="./flowers/" , type=str, help="Path to dataset")
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    parser.add_argument('--arch', dest='arch', default='vgg13', choices=['vgg13', 'vgg16'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001' , help='Learning rate')
    parser.add_argument('--hidden_units', dest='hidden_units', default='512' , help='Number of hidden units')
    parser.add_argument('--epochs', dest='epochs', default='8')
    parser.add_argument('--gpu', action="store_true", default=True , help='Use GPU if available')
    return parser.parse_args()


def main():
    args = todo_args()
    # TODO: Define your transforms for the training, validation, and testing sets
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    batch_size = 20
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                                transforms.RandomRotation(30),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean,std)])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean,std)])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean,std)])
    # TODO: Load the datasets with ImageFolder
    image_datasets = dict()
    image_datasets['train'] = datasets.ImageFolder(train_dir, transform=train_transforms)
    image_datasets['valid'] = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    image_datasets['test']  = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = dict()
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True)
    dataloaders['valid'] = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size)
    dataloaders['test']  = torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size)
    
    model = getattr(models, args.arch)(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    
    if args.arch == "vgg13":
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(feature_num, 1024)),
                                  ('drop', nn.Dropout(p=0.5)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(1024, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))
    elif args.arch == "vgg16":
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(feature_num, 1024)),
                                  ('drop', nn.Dropout(p=0.5)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(1024, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    epochs = int(args.epochs)
    class_index = image_datasets['train'].class_to_idx
    train_model(model, criterion, optimizer, scheduler, args.epochs , args.gpu)
    model.class_to_idx = class_index
    save_checkpoint(model, optimizer, args, classifier)


if __name__ == "__main__":
    main()

