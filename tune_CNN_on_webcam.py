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

from cnnasl import CNN

"""
We apply transfer learning on the model on additional webcam data due to differences in the online
Kaggle dataset and my system's webcam input. This may be due to file type, video data type,
etc. Fine tuning allowed for the model to make accurate predictions on live webcam input.
"""

if __name__ == "__main__":

    kaggle_X = pickle.load(open('alphabet_X_color.sav', 'rb'))
    webcam_X = pickle.load(open("webcam_data_X.sav", 'rb'))
    # print(np.shape(data))
    kaggle_y = pickle.load(open('alphabet_y_color.sav', 'rb'))
    webcam_y = pickle.load(open("webcam_data_y.sav", 'rb'))
    # print(np.shape(target))

    kaggle_X_train, kaggle_X_test, kaggle_y_train, kaggle_y_test = train_test_split(kaggle_X, kaggle_y, test_size=0.3, random_state=77)
    kaggle_y_train = to_categorical(kaggle_y_train, 29)
    kaggle_y_test = to_categorical(kaggle_y_test, 29)

    webcam_X_train, webcam_X_test, webcam_y_train, webcam_y_test = train_test_split(webcam_X, webcam_y, test_size=0.3, random_state=77)
    webcam_y_train = to_categorical(webcam_y_train, 29)
    webcam_y_test = to_categorical(webcam_y_test, 29)

    X_train = np.concatenate((kaggle_X_train, webcam_X_train))
    y_train = np.concatenate((kaggle_y_train, webcam_y_train))
    X_test = np.concatenate((kaggle_X_test, webcam_X_test))
    y_test = np.concatenate((kaggle_y_test, webcam_y_test))

    print("Done loading data!")

    # here we try transfer learning
    # aka freeze the convolutional layers on the base model and only retrain the base layer

    base_model = keras.models.load_model('CNN_on_ASL_alphabet')
    for layer in base_model.layers[:7]:
        layer.trainable = False

    base_model.summary()

    base_model.compile(optimizer='adam',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=2)
    base_model.fit(webcam_X_train, webcam_y_train, epochs=99, validation_data=(X_test, y_test), callbacks=[early_stop])

    est_loss, train_acc = base_model.evaluate(X_train, y_train, verbose=2)
    print("Estimated loss: ", est_loss)
    print("Training accuracy: ", train_acc)

    est_loss, test_acc_webcam = base_model.evaluate(webcam_X_test, webcam_y_test, verbose=2)
    print("Testing accuracy on webcam: ", test_acc_webcam)

    est_loss, test_acc_kaggle = base_model.evaluate(kaggle_X_test, kaggle_y_test, verbose=2)
    print("Testing accuracy on kaggle: ", test_acc_kaggle)

    base_model.save('Fine Tuned ASL CNN')


