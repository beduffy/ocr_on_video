"""
Script reads a video like this
(https://www.youtube.com/watch?v=eqZxBgU-mYY&feature=youtu.be&ab_channel=Kendra%27sLanguageSchool)
and extracts unique frames (between shot transitions) and then runs pytesseract on them to get the
text since I'm lazy with taking down all 1000 phrases in a 3.5 hour video.
"""


import numpy as np
import cv2
import pytesseract
import matplotlib.pyplot as plt
import pandas as pd

filepath = '/home/beduffy/Videos/1000_german_phrases.mp4'

cap = cv2.VideoCapture(filepath)

unique_frames = []
all_pixel_changes = []
all_text = []

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# cap.set(2, 340)  # after intro. didn't work
ret, prev_frame = cap.read()
frame_num = 1

# set to high if we are bigger than 50k for over 2 frames, low if below 50k
# low = True
change_detected_within_x_frames = False
x_frames = 50
last_frame_num_change_detected = -500
last_text = ""

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_num += 1
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if frame_num > 340:
        # Our operations on the frame come here
        diff_image = np.zeros_like(frame)
        cv2.absdiff(frame, prev_frame, diff_image)

        num_pixel_changes = diff_image[diff_image.nonzero()].shape[0]
        all_pixel_changes.append(num_pixel_changes)

        print('frame_num: {}. num_pixel_changes: {}'.format(frame_num, num_pixel_changes))

        if not change_detected_within_x_frames and all([x > 50000 for x in all_pixel_changes[-2:]]):
            print('Change detected. Last two values: {}'.format(all_pixel_changes[-2:]))

            text = pytesseract.image_to_string(frame)
            # print(text)

            if text == last_text:
                print('Text is same as last text, therefore not a real change')
                change_detected_within_x_frames = False
            else:
                print('Current text: {}. \nLast text: {}'.format(text, last_text))
                print('Text is different, a true change')
                change_detected_within_x_frames = True
                last_frame_num_change_detected = frame_num
                last_text = text
                all_text.append(text)

        elif frame_num - last_frame_num_change_detected > x_frames:
            change_detected_within_x_frames = False

        # print('Num unique frames: {}'.format(len(unique_frames)))

        # Display the frame
        wait_length = 1
        # if change_detected_within_x_frames:
        #     # wait_length = 1000
        #     frame_with_text = frame.copy()
        #     cv2.putText(frame_with_text, 'CHANGE DETECTED', (0, 140), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 2,
        #                 cv2.LINE_AA)
        # else:
        #     frame_with_text = frame
        #     # wait_length = 1
        #     pass

        # cv2.imshow('frame', frame_with_text)
        k = cv2.waitKey(wait_length)
        if k == ord('q'):
            break
        if frame_num > 100000:
            print('frame_num was bigger than 10,000, breaking')
            cap.release()
            cv2.destroyAllWindows()
            break

    prev_frame = frame

print('Num unique frames: {}'.format(len(unique_frames)))

# plt.plot(range(len(all_pixel_changes)), all_pixel_changes)
# plt.show()

# all_text_dict = {'text': t for t in all_text}
# df = pd.DataFrame(all_text_dict, index='text')  # columns=['text'],
df = pd.Series(all_text)
print(df.shape)
df.drop_duplicates(inplace=True)  # really solves all the problems I had of finding the shot transition... :P
print(df.shape)
# todo fuzzy score?

df.to_csv('all_text.csv')

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
