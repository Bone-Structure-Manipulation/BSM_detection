import tensorflow as tf
import os
os.sys.path
!pip install opencv-python
import cv2
import sys


# Get user supplied values
imagePath = "abba.png"
cascPath = "haarcascade_frontalface_default.xml"
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)

for (x, y, w, h) in faces:
    crop_img = image[ y:y+h, x:x+w ]
    cv2.imshow("cropped pic",crop_img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()