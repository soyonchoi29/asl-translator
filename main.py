import numpy as np
import statistics as st
import matplotlib.pyplot as plt
import os

import cv2
import time

from keras import models

import handTracker
import spellcheck


def predict(cropped):
    # plt.imshow(cropped)
    # plt.show()

    if cropped.any() >= 1:

        cropped = np.asarray(cropped).reshape((-1, 64, 64, 3))

        predictions = loaded_model.predict(cropped).ravel()
        predicted_letter = letters[np.argmax(predictions)]

        # print(predictions)
        probability = max(predictions) * 100
        # print('confidence:', probability)

        return predicted_letter, probability


if __name__ == '__main__':

    # imgdir = 'C:/Users/soyon/Documents/Codes/ASL-Alphabet-Recognition/dataset/train'
    # letters = sorted(os.listdir(imgdir))
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

    loaded_model = models.load_model('Transfer Learning ASL CNN')

    cap = cv2.VideoCapture(0)
    tracker = handTracker.HandTracker()

    curr_text = ""
    last_letters = []

    # viddir = 'C:/Users/soyon/Documents/Codes/ASL-Translator/Model Live Test Videos'
    # os.chdir(viddir)

    # video_name = 'mp_proc_kaggle_model_live_test.avi'
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(video_name, fourcc, 20.0, (640, 480))

    time_counter = 0

    while True:
        success, image = cap.read()
        frame = cv2.flip(image, 1)

        frame = tracker.find_hands(frame)
        tracker.draw_borders(frame)
        cv2.rectangle(frame, (0, len(frame)),
                             (len(frame[0]), len(frame)-50),
                             (0, 0, 0), -1)

        cv2.putText(frame, '{}: {}'.format('Translated text', curr_text),
                    (20, len(frame)-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        if len(last_letters) == 10:
            curr_text += st.mode(last_letters)
            print(curr_text)
            last_letters = []

        # out.write(frame)
        if (tracker.results.multi_hand_landmarks):
            cropped_hand_img = tracker.slice_hand_imgs(cv2.flip(image, 1), 0, 100)
            result = predict(cropped_hand_img)
            if result:
                prediction, prob = result
                prob = round(prob, 2)
                if prob >= 25:
                    tracker.display_letters(frame, 0, prediction, prob)
                    if (time_counter % 15 == 0) and\
                            (prediction != 'del') and (prediction != 'space') and (prediction != 'nothing'):
                        last_letters.append(prediction)
                        print(last_letters)

        cv2.imshow("Signed English Translator", frame)
        k = cv2.waitKey(1)

        if (k == 8):
            curr_text = curr_text[:-1]
            print("Deleted letter!")
        elif (k == ord(' ')):
            curr_text = spellcheck.spellCheck(curr_text)
            curr_text += ' '
            print("Added space!")

        if cv2.getWindowProperty("Signed English Translator", cv2.WND_PROP_VISIBLE) < 1:
            break

        time_counter = time_counter + 1

    # out.release()
    cv2.destroyAllWindows()

