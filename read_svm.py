#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


# In[2]:


class CustomImageFolder(Dataset):
    def __init__(self, train_1_dir, train_0_dir, transform=None):
        self.train_1_dir = train_1_dir
        self.train_0_dir = train_0_dir
        self.transform = transform
        self.samples = []

        # Add all images from train_te
        for root, _, files in os.walk(train_1_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self.samples.append((file_path, 1))

        # Add selected images from train_image
        for root, _, files in os.walk(train_0_dir):
            files = np.random.choice(files, 100, replace=False)
            for file in files:
                file_path = os.path.join(root, file)
                self.samples.append((file_path, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class CustomTensorDataset(Dataset):
    def __init__(self, images, labels, image_size=(128, 128)):
        self.images = torch.tensor(images, dtype=torch.float32).view(-1, 1, *image_size) #change the size of array to align with extract_features
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]



# In[3]:


train_1_dir = "../train_1"
test_1_dir = "../test_1"
train_0_dir = "../train_0"
test_0_dir = "../test_0"

# Load the images and data loader for training and testing
Dataset_tr = CustomImageFolder(train_1_dir, train_0_dir, transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
]))
Dataset_te = CustomImageFolder(test_1_dir, test_0_dir, transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
]))

#trainloader = torch.utils.data.DataLoader(dataset=Dataset_tr, batch_size=batch_size, shuffle=True)
#testloader = torch.utils.data.DataLoader(dataset=Dataset_te, batch_size=batch_size, shuffle=True)

