import math
import torch
import torch.nn.functional as F
import data_loader
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import model_loader

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(100)


def check_accuracy_part(loader, model, phase):
    print('Checking accuracy on %s set' % phase)   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return acc

def train_part(model, optimizer, is_inception, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device, dtype = dtype)  # move the model parameters to CPU/GPU
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for e in range(epochs):
        for t, (x, y) in enumerate(dataloaders["train"]):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            if is_inception:
                outputs, aux_outputs = model(x)
                loss1 = F.cross_entropy(outputs, y)
                loss2 = F.cross_entropy(aux_outputs, y)
                loss = loss1 + 0.4*loss2
            else:
                outputs = model(x)
                loss = F.cross_entropy(outputs, y)
            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

        
        print('epoche %d, loss = %f' % (e, loss.item()))
        train_acc = check_accuracy_part(dataloaders["train"], model, "train")
        test_acc = check_accuracy_part(dataloaders["val"], model, "test")
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        print()
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# VGG16_bn model
# vgg16 = models.vgg16_bn(pretrained=True)
# for param in vgg16.parameters():
#     param.requires_grad = False
# vgg16.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
#                                        torch.nn.ReLU(),
#                                        torch.nn.Dropout(p=0.8),
#                                        torch.nn.Linear(4096, 20))                                                                  
# vgg16 = vgg16.to(device, dtype=dtype)
# optimizer = torch.optim.Adam(vgg16.parameters(), lr = 0.001)
# train_part(vgg16, optimizer, epochs=20)
if __name__ == "__main__":
    
    dtype = torch.float32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    feature_extract = True
    model, input_size = model_loader.initialize_model(model_name = "resnet152", num_classes = 20, feature_extract = feature_extract, use_pretrained=True)
    params_to_update = model.parameters()
    print("Params to learn:")

    # display parameters
    # for name, param in model.named_parameters():
    #     print(name)

    

    param_to_update_name_prefix = ["layer4"]
    for name, param in model.named_parameters():
        for i in param_to_update_name_prefix:
            if i in name:
                param.requires_grad = True

    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

# load data
# data_set = data_loader.data_loader(input_size, input_size, False)
# train_data, train_label = data_set.get_train_data()
# test_data, test_label = data_set.get_test_data()

# train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

# test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomRotation(degrees = 10),
            # transforms.RandomResizedCrop((224, 224)),
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3), value= "random", inplace=False),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'sorted_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                                shuffle=True, num_workers=4, pin_memory=True)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes




    # Get a batch of training data
    # inputs, classes = next(iter(dataloaders['train']))
    # print(input, classes)
    # Make a grid from batch
    # out = torchvision.utils.make_grid(inputs)

    # imshow(out, title=[class_names[x] for x in classes])
    


    optimizer = torch.optim.Adam(params_to_update, lr = 0.0001)
    # first train fc layer
    model = train_part(model, optimizer, False, epochs=30)
    # fine tune the whole model
    # for param in model.parameters():
        # param.requires_grad = True
    # model = train_part(model, optimizer, False, epochs=10)