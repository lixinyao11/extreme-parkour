
import torchvision
from torchvision import models
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset
import torch
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, all_dataset):
        self.data = list(all_dataset.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, lidar_data = self.data[idx]
        image_tensor = torch.from_numpy(image).float()  # Assuming image is a numpy array
        lidar_tensor = torch.from_numpy(np.array(lidar_data)).float()  # Assuming lidar_data is a numpy array
        return image_tensor, lidar_tensor


class ResNetModel(torch.nn.Module):
    def __init__(self, resnet_type):
        super(ResNetModel, self).__init__()
        if resnet_type == 'resnet18':
            self.resnet = models.resnet18(weights='IMAGENET1K_V1')
            print("Using resnet18")

        elif resnet_type == 'resnet34':
            self.resnet = models.resnet34(weights='IMAGENET1K_V1')
            print("Using resnet34")
        else:
            raise NotImplementedError
        self.num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(self.num_ftrs, 11)

    def forward(self, x):
        x_1 = self.resnet(x)
        return x_1