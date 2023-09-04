import keras.models
import numpy as np
import pandas as pd
import os
import cv2

from skimage.transform import resize

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras import losses
from keras.datasets import cifar10
from keras import layers, models
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

import pickle
import gc


imgdir = 'C:/Users/soyon/Documents/Codes/ASL-Alphabet-Recognition/dataset/train'
testdir = 'C:/Users/soyon/Documents/Codes/ASL-Alphabet-Recognition/webcam dataset'
letters = sorted(os.listdir(imgdir))


if __name__ == "__main__":

    # model = models.load_model('CNN_on_ASL_alphabet')
    # model = models.load_model('ASL.h5')
    model = models.load_model('CNN on webcam')

    count = 0
    label = 0

    X = pickle.load(open('alphabet_X_color.sav', 'rb'))
    # print(np.shape(self.X))
    y = pickle.load(open('alphabet_y_color.sav', 'rb'))
    # print(self.y)
    # print(np.shape(self.y))
    y = to_categorical(y, 29)

    print("Done loading testing data!")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

    est_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print("Estimated loss: ", est_loss)
    print("Testing accuracy: ", test_acc)

