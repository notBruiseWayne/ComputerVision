import os
import cv2 as cv
import numpy as np
import time

start_time = time.time()
p = []
for i in os.listdir(r'E:\cv01\Photos\Faces\Train'):
    p.append(i)
print(p)
DIR = r'E:\cv01\Photos\Faces\Train'
haar_face = cv.CascadeClassifier(r'haar_face.xml')
features = []
labels = []


def create_train():
    for person in p:
        subFolder = os.path.join(DIR, person)
        label = p.index(person)
        for image in os.listdir(subFolder):
            image_path = os.path.join(subFolder, image)
            image_array = cv.imread(image_path)
            gray = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y + h, x:x + w]
                features.append(faces_roi)
                labels.append(label)


create_train()
features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features,labels)
np.save('features.npy', features)
np.save('labels.npy', labels)
face_recognizer.save(r'E:\cv01\face_trained.yml')
end_time = (time.time()) - start_time
print(f'Training done-----------\nTotal {end_time} seconds took for the process to finish!')
