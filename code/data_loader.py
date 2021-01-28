import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import transforms
import torch
class data_loader:
    def __init__(self, x, y, using_pretrained):
        super().__init__()
        test = pd.read_csv("data/annotation.csv")
        train = pd.read_csv("data/training.csv")
        self.train_dict = {}
        self.test_dict = {}
        for i in range(len(test["FileID"])):
            self.test_dict[test["FileID"][i]] = test["SpeciesID"][i]
        for i in range(len(train["FileID"])):
            self.train_dict[train["FileID"][i]] = train["SpeciesID"][i]
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        if using_pretrained:
            mean = [0.485, 0.456, 0.406]
            std  = [0.229, 0.224, 0.225]
            transform = transforms.Compose([
                transforms.Resize((x, y)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((x, y)),
                transforms.ToTensor(),
            ])
        for addr, label in self.train_dict.items():
            path = os.path.join("data\data", addr + ".jpg")
            t = transform(Image.open(path))
            train_data.append(t)
            train_label.append(label)
            
        for addr, label in self.test_dict.items():
            path = os.path.join("data\data", addr + ".jpg")
            t = transform(Image.open(path))
            test_data.append(t)
            test_label.append(label)

        self.train_data = torch.stack(train_data, dim=0)
        self.train_label = torch.from_numpy(np.array(train_label))
        self.test_data = torch.stack(test_data, dim=0)
        self.test_label = torch.from_numpy(np.array(test_label))
    def get_train_data(self):
        return self.train_data, self.train_label

    def get_test_data(self):
        return self.test_data, self.test_label

       
        
            
        
if __name__ == "__main__":
    test = data_loader(224, 224)
    
    
        
    