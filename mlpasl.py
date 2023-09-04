import numpy as np
import pandas as pd
import os
import cv2
import skimage

from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray

import matplotlib.pyplot as plt

import sklearn.neural_network._multilayer_perceptron as mlp
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

from keras.utils import to_categorical


imgdir = 'C:/Users/soyon/Documents/Codes/ASL-Alphabet-Recognition/dataset/train'
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

                if imgind <= 1300:
                    continue
                elif imgind >= 1600:
                    break

                img = imread(datadir + '/' + folder + '/' + image)
                img = resize(img, (64, 64))
                img = rgb2gray(img)
                img /= 255

                self.X.append(img.flatten())
                self.y.append(index)

            index += 1

        self.X = np.array(self.X)
        # print(np.shape(self.X))
        self.y = np.array(self.y)
        # print(self.y)
        # print(np.shape(self.y))

        return self.X, self.y


def MLP(X, y):
    # data is a dictionary
    model = mlp.MLPClassifier()

    n = len(data)
    # print("n=", n)

    model.fit(X, y)

    return model


if __name__ == '__main__':

    myData = Data()
    data, target = myData.load_data(imgdir)
    print("Done loading data!")

    # print(target)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, shuffle=True)

    # y_train = to_categorical(y_train)

    my_model = MLP(X_train, y_train)
    predicted = my_model.predict(X_test)

    # pred_nums = []
    pred_letters = []
    for i in range(len(predicted)):
        # pred_nums.append(np.argmax(predicted[i]))
        pred_letters.append(letters[predicted[i]])

    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, pred_letters):
        ax.set_axis_off()
        image = image.reshape(64, 64)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

    plt.show()

    print("Accuracy: ", accuracy_score(predicted, y_test)*100)

