import cv2 as cv
import numpy as np

# img = cv.imread(r'Photos/people.jpg')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('person', gray)
# haar = cv.CascadeClassifier(r'haar_face.xlm')
# faces_rect = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
# print(f'Faces Detected in image = {len(faces_rect)}')
# for (x, y, h, w) in faces_rect:
#     cv.rectangle(img, (x, y), (x + w, y + h), (125, 255, 45), thickness=2)
# cv.imshow('Faces Detected', img)
# cv.waitKey(0)

haar_face = cv.CascadeClassifier(r'haar_face.xml')
haar_eye = cv.CascadeClassifier(r'haar_eye.xml')


def detect(gr, fa):
    faces = haar_face.detectMultiScale(gr, scaleFactor=1.3, minNeighbors=10)
    for (x, y, w, h) in faces:
        cv.rectangle(fa, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
        cropped_gray = gr[y:y + h, x:x + h]
        cropped_frame = fa[y:y + h, x:x + h]
        eyes = haar_eye.detectMultiScale(cropped_gray, scaleFactor=1.3, minNeighbors=2)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(cropped_frame, (ex, ey), (ex + w, ey + h), (0, 0, 255), thickness=1)
    return fa


capture = cv.VideoCapture(r'yourPath\jimi.mp4')
while True:
    isTrue, frame = capture.read()  # gives two values  a boolean for the status of read and the frame of the video that
    # can be shown with cv.imshow() method
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv.imshow('Video', canvas)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break
capture.release()
cv.destroyAllWindows()
