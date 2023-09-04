import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import matplotlib.pyplot as plt
import keras.models as models
import pickle
import numpy as np
import cv2
from skimage.transform import resize


imgdir = 'C:/Users/soyon/Documents/Codes/ASL-Alphabet-Recognition/dataset/train'
webcamdir = 'C:/Users/soyon/Documents/Codes/ASL-Alphabet-Recognition/webcam dataset'
testdir = 'C:/Users/soyon/Documents/Codes/ASL-Alphabet-Recognition/dataset/test'
letters = sorted(os.listdir(webcamdir))


def load_data(datadir):

    X = np.empty((461, 64, 64, 3), dtype=np.float32)
    y = np.empty((461,), dtype=int)

    label = 0
    count = 0

    folders = sorted(os.listdir(datadir))
    labels = folders
    # print(folders)

    # separate folder for each letter
    for folder in folders:

        print("Loading images from folder", folder, "has started.")

        imgind = -1

        for image in os.listdir(datadir + '/' + folder):

            imgind += 1
            if imgind >= 10:
                break

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

                X[count] = img
                y[count] = label

                count += 1

        label += 1

    X = np.array(X)
    print(np.shape(X))
    y = np.array(y)
    # print(y)
    print(np.shape(y))

    return X, y

def plot_heatmap(y_true, y_pred, class_names, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        square=True,
        xticklabels=class_names,
        yticklabels=class_names,
        fmt='d',
        cmap=plt.cm.Blues,
        cbar=False,
    )
    plt.suptitle(title, fontsize=18)
    # plt.xticks(rotation=45, ha='right')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)


if __name__ == '__main__':

    # webcam_X, webcam_true_y = load_data(webcamdir)

    webcam_model = models.load_model('CNN on webcam')
    webcam_X = pickle.load(open('webcam_data_X.sav', 'rb'))
    webcam_true_y = pickle.load(open('webcam_data_y.sav', 'rb'))
    webcam_pred_y = np.argmax(webcam_model.predict(webcam_X), axis=1)
    # print(webcam_pred_y)

    # kaggle_model = models.load_model('CNN_on_ASL_alphabet')
    # kaggle_X = np.array(pickle.load(open('alphabet_X_color.sav', 'rb')))
    # kaggle_true_y = np.array(pickle.load(open('alphabet_y_color.sav', 'rb')))
    # kaggle_pred_y = np.argmax(kaggle_model.predict(webcam_X), axis=1)
    #
    # kaggle_fine_tuned = models.load_model('Fine Tuned ASL CNN')
    # tuned_true_y = np.array(np.concatenate([(webcam_true_y, kaggle_true_y)]))
    # pred1 = np.array(np.argmax(kaggle_fine_tuned.predict(webcam_X)))
    # pred2 = np.array(np.argmax(kaggle_fine_tuned.predict(kaggle_X)))
    # tuned_pred_y = np.argmax(kaggle_fine_tuned.predict(webcam_X), axis=1)

    plot_heatmap(webcam_true_y, webcam_pred_y, letters, title='Webcam CNN Confusion Matrix')
    # plot_heatmap(webcam_true_y, kaggle_pred_y, letters, ax2, title='Kaggle CNN')
    # plot_heatmap(webcam_true_y, tuned_pred_y, letters, ax3, title='Fine-tuned CNN')

    # plt.tight_layout()
    plt.show()

