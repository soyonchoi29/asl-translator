import numpy
import pandas as pd
import os

import matplotlib.pyplot as plt
import sklearn.neural_network._multilayer_perceptron as mlp
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def MLP():
    # data is a dictionary
    data = load_digits()
    model = mlp.MLPClassifier()

    n = len(data)
    # print("n=", n)

    X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=0.3, shuffle=False)

    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

    plt.show()

    return model


if __name__ == "__main__":
    mlp = MLP()

