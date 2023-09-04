import numpy as np
import pandas as pd
import os
import cv2
import skimage

from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import flatten

import torchvision
from torchvision import transforms, datasets
from torchsummary import summary

import pickle
import gc


imgdir = 'C:/Users/soyon/Documents/Codes/ASL-Alphabet-Recognition/dataset/train'
testdir = 'C:/Users/soyon/Documents/Codes/ASL-Alphabet-Recognition/dataset/test'
letters = sorted(os.listdir(imgdir))


class Data:

    def __init__(self):
        self.X, self.y = [], []
        self.labels = []

    def load_data(self, datadir):

        index = 0

        folders = sorted(os.listdir(datadir))
        images = []
        self.labels = folders
        # print(folders)

        # separate folder for each letter
        for folder in folders:

            print("Loading images from folder", folder, "has started.")
            imgind = -1

            for image in os.listdir(datadir + '/' + folder):

                imgind += 1
                # print(imgind)

                # if imgind <= 1300:
                #     continue
                # elif imgind >= 1600:
                #     break

                img = imread(datadir + '/' + folder + '/' + image)
                img = rgb2gray(img)
                img = resize(img, (64, 64, 1))
                img /= 255

                self.X.append(img)
                self.y.append(index)

            index += 1

        self.X = np.array(self.X)
        print(np.shape(self.X))
        self.y = np.array(self.y)
        # print(self.y)
        # print(np.shape(self.y))

        return self.X, self.y


class LeNet(nn.Module):
    def __init__(self, numChannels, classes):

        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=20, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.fc1 = nn.Linear(in_features=800, out_features=500)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=500, out_features=classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        output = self.logSoftmax(x)

        return output


def save(elem, dir):
    pickle.dump(elem, open(dir, 'wb'))
    print("** Current model has been saved! **")


def load(dir):
    pickle.load(open(dir, 'rb'))
    print("** Model was loaded! **")


if __name__ == "__main__":

    train_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

    train_dataset = datasets.ImageFolder(imgdir, transforms=train_transforms)
    train_sample_size = len(train_dataset)
    train_dataset

    test_dataset = datasets.ImageFolder(testdir, transforms=test_transforms)
    test_dataset

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=32,
                                                   shuffle=True,
                                                   num_workers=4)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=32,
                                                  shuffle=False,
                                                  num_workers=4)

    print("Done loading data!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device

    model = torchvision.models.resnet50(pretrained=True)
    model

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, len(letters))
    model

    model.to(device)
    summary(model, (3, 224, 224), batch_size=2)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # new_cnn_class.print_acc()
    # new_cnn_class.save_model("CNN_on_ASL_alphabet")


