import math
import torch
import torch.nn.functional as F
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
import logging
import pandas as pd
from PIL import Image
import sys
def loader():
    transforms_for_test_img = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_file = pd.read_csv("/cache/af2020cv-v5-formal/test.csv")
    for i in test_file["FileID"]:
        im = Image.open(os.path.join("/cache/af2020cv-v5-formal/data", i + ".jpg"))
        im_after = transforms_for_test_img(im)
        yield im_after, i



logging.basicConfig(filename = "result.log", level=logging.NOTSET)

def run_on_test(device, dtype, loader, model):
    print('Checking accuracy on test set')
    model.eval()  # set model to evaluation mode
    rec_file_id = []
    rec_label = []
    with torch.no_grad():
        for x, i in loader:
            x = torch.unsqueeze(x, dim=0)
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            scores = model(x)
            _, preds = scores.max(1)
            label = preds[0].item()
            rec_file_id.append(i)
            rec_label.append(label)
    df = pd.DataFrame({'FileID':rec_file_id, 'SpeciesID':rec_label})
    df.to_csv("results.csv",index=False, sep=',')
    

if __name__ == "__main__":
    device = "cuda:0"
    dtype = torch.float32
    model = torch.load("./model.pkl")
    run_on_test(device = device, dtype = dtype, loader = loader, model = model)
