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


imgdir = 'C:/Users/soyon/Documents/Codes/ASL-Alphabet-Recognition/webcam dataset'
letters = sorted(os.listdir(imgdir))


class Data:

    def __init__(self):
        self.X = np.empty((650, 64, 64, 3), dtype=np.float32)
        self.y = np.empty((650,), dtype=int)
        self.labels = []

    def load_data(self, datadir):

        label = 0
        count = 0

        folders = sorted(os.listdir(datadir))
        self.labels = folders
        # print(folders)

        # separate folder for each letter
        for folder in folders:

            print("Loading images from folder", folder, "has started.")

            for image in os.listdir(datadir + '/' + folder):

                img = cv2.imread(datadir + '/' + folder + '/' + image)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = resize(img, (64, 64, 3))
                    img = np.asarray(img).reshape((-1, 64, 64, 3))
                    # img = img * (.1/255)

                    # plt.imshow(img)
                    # plt.show()

                    # print(img)
                    # print(np.size(img))

                    self.X[count] = img
                    self.y[count] = label

                    count += 1

            label += 1

        self.X = np.array(self.X)
        # print(np.shape(self.X))
        self.y = np.array(self.y)
        # print(self.y)
        # print(np.shape(self.y))

        return self.X, self.y


class CNN:

    def __init__(self, train_images, train_labels, test_images, test_labels):
        self.model = models.Sequential()
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels

    def build_model(self):

        # create convolutional base
        self.model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(64, 64, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        # add dense layers
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dense(29, activation='softmax'))
        self.model.summary()

        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        return self.model

    def fit_model(self, num_epochs):

        early_stop = EarlyStopping(monitor='val_loss', patience=2)
        self.model.fit(self.train_images, self.train_labels, epochs=num_epochs,
                       batch_size=64, verbose=2,
                       validation_data=(self.test_images, self.test_labels),
                       callbacks=[early_stop])

        return self.model

    def print_acc(self):
        est_loss, train_acc = self.model.evaluate(self.train_images, self.train_labels, verbose=2)
        print("Estimated training loss: ", est_loss)
        print("Training accuracy: ", train_acc)

        est_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels, verbose=2)
        print("Estimated loss: ", est_loss)
        print("Testing accuracy: ", test_acc)

    def save_model(self, dir):
        self.model.save(dir)
        print("** Current model has been saved! **")

    def load_model(self, dir):
        self.model = models.load_model(dir)
        print("** Model was loaded! **")

    def update_size(self, index, dimensions):
        self.model.layers[index].input_shape = dimensions
        return self.model


if __name__ == "__main__":

    # myData = Data()
    # data, target = myData.load_data(imgdir)
    #
    # pickle.dump(data, open("webcam_data_X.sav", 'wb'))
    # pickle.dump(target, open("webcam_data_y.sav", 'wb'))

    data = pickle.load(open("webcam_data_X.sav", 'rb'))
    # print(np.shape(data))
    target = pickle.load(open("webcam_data_y.sav", 'rb'))
    # print(np.shape(target))

    print("Done loading data!")

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=77)
    y_train = to_categorical(y_train, 29)
    print(np.shape(y_train))
    y_test = to_categorical(y_test, 29)
    print(np.shape(y_test))

    # del data
    # del target
    # gc.collect()

    # X_test = []
    # y_test = []
    # labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 27, 14, 15, 16, 17, 18, 28, 19, 20, 21, 22, 23, 24, 25]
    # index = 0
    # for image in os.listdir(testdir):
    #
    #     img = cv2.imread(testdir + '/' + image)
    #     img = resize(img, (64, 64, 3))
    #     img /= 255
    #     # print(np.size(img))
    #
    #     X_test.append(img)
    #     y_test.append(labels[index])
    #     index += 1
    #
    # X_test = np.asarray(X_test)
    # y_test = np.asarray(y_test)
    # y_test = to_categorical(y_test, 29)
    #
    # print("Done loading testing data!")

    # new_cnn_class = CNN(X_train, y_train, X_test, y_test)
    # new_cnn_class.build_model()
    # new_cnn_model = new_cnn_class.fit_model(num_epochs=99)
    #
    # new_cnn_class.print_acc()
    #
    # new_cnn_class.save_model("CNN on webcam")

    loaded_model = keras.models.load_model('CNN on MediaPipe Processed Kaggle')

    # kaggle_X = pickle.load(open('alphabet_X_color.sav', 'rb'))
    # kaggle_y = pickle.load(open('alphabet_y_color.sav', 'rb'))
    target = to_categorical(target, 29)
    est_loss, test_acc = loaded_model.evaluate(data, target, verbose=2)
    print("Estimated loss: ", est_loss)
    print("Testing accuracy: ", test_acc)

