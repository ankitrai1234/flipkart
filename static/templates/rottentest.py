import cv2
import os
import time
from time import sleep
from lobe import ImageModel
previous ="unkno"

video_capture = cv2.VideoCapture(0)

model = ImageModel.load('/home/pi/test TensorFlow')

file = '/home/pi/lobe-python/rottenapple.png'
process_this_frame = True

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if process_this_frame:
        
        cv2.imwrite(file, small_frame)
        result = model.predict_from_file('/home/pi/lobe-python/rottenapple.png')
        print(result)
    process_this_frame = not process_this_frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
