import numpy
import pandas as pd
import os
import matplotlib.pyplot as plt

from keras import losses
from keras.datasets import cifar10
from keras import layers, models


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

train_images = X_train/255.0
train_labels = y_train
test_images = X_test/255.0
test_labels = y_test


class CNN:

    def __init__(self):
        self.model = models.Sequential()

    def build_model(self):

        # create convolutional base
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        # add dense layers
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10, activation='softmax'))
        self.model.summary()

        self.model.compile(optimizer='adam',
                           loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        return self.model

    def fit_model(self):
        self.model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
        self.model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
        self.model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
        return self.model

    def print_acc(self):
        est_loss, test_acc = self.model.evaluate(test_images, test_labels, verbose=2)
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


    def update_size(self, index, dimensions):
        self.model.layers[index].input_shape = dimensions
        return self.model


    def update_size(self, index, dimensions):
        self.model.layers[index].input_shape = dimensions
        return self.model


if __name__ == "__main__":

    new_cnn_class = CNN()
    new_cnn_class.build_model()
    new_cnn_model = new_cnn_class.fit_model()

    new_cnn_class.print_acc()

    new_cnn_class.save_model("CNN_on_CIFAR10")  # loss = 1.298, acc = 0.5015

    # mnist_cnn_class = CNN()
    # mnist_cnn_class.load_model("CNN_on_MNIST")
    # updated_model = mnist_cnn_class.fit_model()
    #
    # mnist_cnn_class.print_acc()
