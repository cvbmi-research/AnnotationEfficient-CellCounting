import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from VGG_image import *
import torchvision.transforms.functional as F

class listDataset1(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False):
        if train:
            root = root * 4 
        random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        
        
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        img_path = self.lines[index]
        img, target, cluster_center = load_labeled_data(img_path,self.train) #labeled samples
      
        if self.transform is not None:
            img = self.transform(img)
        return img, target, cluster_center