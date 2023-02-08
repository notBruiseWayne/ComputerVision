import cv2 as cv
import numpy as np

people = ['Amanda Seyfried', 'Anne Hathaway', 'Mark Wahlberg', 'Matt Damon', 'Pedro Pascal', 'Rachel Mcadams']
haar_face = cv.CascadeClassifier(r'haar_face.xml')
# features = np.load('features.npy')
# labels = np.load('labels.npy')
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'E:\cv01\Photos\Faces\validation\amanda2.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Image', gray)
faces_rect = haar_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)
for (x, y, h, w) in faces_rect:
    faces_roi = gray[y:y + h, x:x + w]
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label={people[label]} with the confidence of {confidence}')
    cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), thickness=1)
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=1)
cv.imshow('Detected img', img)
cv.waitKey(0)
