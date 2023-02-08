# ComputerVision
CV2 projects and models
Face_recog.py and face_train.py uses haar cascades (face and eye) algorithms for face detection,
The detectMultiScale method is only able to detect faces that are upfront, side poses and other angles are not detected as a face, also requires tuning hyperparameters
i.e scaleFactor and minNeighbor for better results, works better on smaller resolution images/videos.
And opencv's face recognizer LBPHFaceRecognizer to train on images 
and for predictions on the images.
The model is somewhat accurate provided it was trained on only a few (less than 100) images
