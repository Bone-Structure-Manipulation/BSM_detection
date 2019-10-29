import cv2
import numpy as np
import dlib
import face_recognition
import os

font=cv2.FONT_HERSHEY_SIMPLEX

known_image = face_recognition.load_image_file("keira.jpg")
unknown_image = face_recognition.load_image_file("natalie.jpg")
MODEL_MEAN_VALUES = (78.42633776603, 87.7689143744, 114.895847746)

age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']


def load_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt','age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
    return age_net, gender_net

def video_detector(age_net, gender_net):
    font = cv2.FONT_HERSHEY_SIMPLEX

if __name__ == "__main__":
        age_net, gender_net = load_caffe_models()
        video_detector(age_net, gender_net)

known_image = cv2.resize(known_image, dsize=(0, 0), fx=1, fy=1,interpolation=cv2.INTER_AREA)
unknown_image = cv2.resize(unknown_image, dsize=(0, 0), fx=1, fy=1, interpolation=cv2.INTER_AREA)
biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
known_gray = cv2.cvtColor(known_image, cv2.COLOR_BGR2GRAY)
unknown_gray = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2GRAY)
known_faces = face_cascade.detectMultiScale(known_gray, 1.1, 5)
unknown_faces = face_cascade.detectMultiScale(unknown_gray, 1.1, 5)

if(len(known_faces)>0):
    print("Found {} faces".format(str(len(known_faces))))

if(len(unknown_faces)>0):
    print("Found {} faces".format(str(len(unknown_faces))))

#known_age_gender
for(x, y, w, h)in known_faces:
    #Get Face
    cv2.rectangle(known_image, (x, y), (x+w, y+h), (255,255,0),2)
    face_img = known_image[y:y+h, h:h+w].copy()
    blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    #Predict Gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    known_gender = gender_list[gender_preds[0].argmax()]
    print("Gender : " + known_gender)
    #Predict Age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    known_age = age_list[age_preds[0].argmax()]
    print("Age Range : " + known_age)


    known_overlay_text = "{}_{}".format(known_gender, known_age)
    cv2.putText(known_image, known_overlay_text, (x,y), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

#unknown_age_gender
for (x, y, w, h) in unknown_faces:
    # Get Face
    cv2.rectangle(unknown_image, (x, y), (x + w, y + h), (255, 255, 0), 2)
    face_img = unknown_image[y:y + h, h:h + w].copy()
    blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    # Predict Gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    unknown_gender = gender_list[gender_preds[0].argmax()]
    print("Gender : " + unknown_gender)
    # Predict Age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    unknown_age = age_list[age_preds[0].argmax()]
    print("Age Range : " + unknown_age)



    results = face_recognition.compare_faces([biden_encoding], unknown_encoding)



    unknown_overlay_text = "{}_{}".format(unknown_gender, unknown_age)
    results_text = "{}".format(results)

    b, g, r = cv2.split(known_image)
    known_image = cv2.merge([r,g,b])
    b, g, r = cv2.split(unknown_image)
    unknown_image = cv2.merge([r,g,b])

    cv2.putText(unknown_image, unknown_overlay_text, (x, y), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(unknown_image, results_text, (10, 50), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(known_image, results_text, (10, 50), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('known', known_image)
    cv2.imshow('unknown', unknown_image)
    cv2.waitKey(0)