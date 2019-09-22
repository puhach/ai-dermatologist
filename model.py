import numpy as np
import torch
from torchvision import datasets, transforms

input_size = 224

preprocessing_train = transforms.Compose([
                                        transforms.RandomRotation(20),
                                        transforms.RandomHorizontalFlip(0.3),
                                        transforms.RandomResizedCrop(input_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
                                        ])

preprocessing_test = transforms.Compose([
                                        transforms.Resize(input_size),
                                        transforms.CenterCrop(input_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])


dataset_train = datasets.ImageFolder(root='data/train', transform=preprocessing_train)
dataset_valid = datasets.ImageFolder(root='data/valid', transform=preprocessing_test)
dataset_test = datasets.ImageFolder(root='data/test', transform=preprocessing_test)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True)
dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=64, shuffle=True)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=64, shuffle=True)

batch = next(iter(dataloader_train))
print(batch)