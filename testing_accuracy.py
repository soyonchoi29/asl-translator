import keras.models
import numpy as np
import pandas as pd
import os
import cv2

from skimage.transform import resize
from skimage.color import rgb2gray

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras import losses
from keras.datasets import cifar10
from keras import layers, models
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

from sklearn import svm, decomposition
from sklearn.metrics import accuracy_score

import pickle
import gc


kaggledir = 'C:/Users/soyon/Documents/Codes/ASL-Alphabet-Recognition/dataset/train'
webcamdir = 'C:/Users/soyon/Documents/Codes/ASL-Alphabet-Recognition/webcam dataset'
datasets = [kaggledir, webcamdir]
webcam_svm_dir = 'C:/Users/soyon/Documents/Codes/ASL-Alphabet-Recognition/saved models/webcam_svm_model.sav'
kaggle_svm_dir = 'C:/Users/soyon/Documents/Codes/ASL-Alphabet-Recognition/saved models/svm_model.sav'


if __name__ == '__main__':
    webcam_svm = pickle.load(open(webcam_svm_dir, 'rb'))
    kaggle_svm = pickle.load(open(kaggle_svm_dir, 'rb'))

    X_tests = np.empty((2, 29, 4096), dtype=np.float32)
    y_tests = np.empty((2, 29,), dtype=int)

    for i in range(len(datasets)):
        label = 0
        count = 0

        folders = sorted(os.listdir(datasets[i]))
        labels = folders
        # print(folders)

        # separate folder for each letter
        for folder in folders:

            print("Loading images from folder", folder, "has started.")

            images = os.listdir(datasets[i] + '/' + folder)

            img = cv2.imread(datasets[i] + '/' + folder + '/' + images[0])
            if img is not None:

                if i == 0:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img = rgb2gray(img)
                img = resize(img, (64, 64))
                img = np.asarray(img).reshape((-1, 64, 64))
                # img = img * (.1/255)

                # plt.imshow(img)
                # plt.show()

                # print(img)
                # print(np.size(img))

                X_tests[i][count] = img.flatten()
                y_tests[i][count] = label

                count += 1

            label += 1

        X_tests[i] = np.array(X_tests[i])
        # print(np.shape(self.X))
        y_tests[i] = np.array(y_tests[i])
        # print(self.y)
        # print(np.shape(self.y))

    kaggle_svm_on_webcam = accuracy_score(y_tests[1], kaggle_svm.predict(X_tests[1]))
    kaggle_svm_on_kaggle = accuracy_score(y_tests[0], kaggle_svm.predict(X_tests[0]))
    webcam_svm_on_webcam = accuracy_score(y_tests[1], webcam_svm.predict(X_tests[1]))
    webcam_svm_on_kaggle = accuracy_score(y_tests[0], webcam_svm.predict(X_tests[0]))

    print('kaggle_svm_on_webcam', kaggle_svm_on_webcam)
    print('kaggle_svm_on_kaggle', kaggle_svm_on_kaggle)
    print('webcam_svm_on_webcam', webcam_svm_on_webcam)
    print('webcam_svm_on_kaggle', webcam_svm_on_kaggle)
