import dlib
import face_recognition
import cv2
import os
import numpy as numpy
known_image = face_recognition.load_image_file("b7.jpg")
unknown_image = face_recognition.load_image_file("a7.jpg")

biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
print(results)