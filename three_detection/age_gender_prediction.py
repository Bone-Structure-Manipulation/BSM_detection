import cv2
import numpy as np
import dlib
import face_recognition
import os

font=cv2.FONT_HERSHEY_SIMPLEX

known_image = face_recognition.load_image_file("person.jpg")

MODEL_MEAN_VALUES = (78.42633776603, 87.7689143744, 114.895847746)

age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']

#

def load_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt','age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
    return age_net, gender_net

def video_detector(age_net, gender_net):
    font = cv2.FONT_HERSHEY_SIMPLEX

if __name__ == "__main__":
        age_net, gender_net = load_caffe_models()
        video_detector(age_net, gender_net)
capture = cv2.VideoCapture(0)#카메라띄우기
capture.set(3, 480)
capture.set(4, 320)

cnt = 0
while True:
    ret, img1 = capture.read()#카메라 캡쳐
    cv2.imshow('cam',img1)

    if cv2.waitKey(1) & 0xFF == ord('q'):

        height, width = img1.shape[:2]
        print('{} x {}'.format(width, height))

        img1 = cv2.resize(img1, dsize=(0, 0), fx=1, fy=1, interpolation=cv2.INTER_AREA)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if(len(faces)>0):
            print("Found {} faces".format(str(len(faces))))

        for(x, y, w, h)in faces:
            #Get Face
            cv2.rectangle(img1, (x, y), (x+w, y+h), (255,255,0),2)
            face_img = img1[y:y+h, h:h+w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            #Predict Gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            print("Gender : " + gender)
            #Predict Age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            print("Age Range : " + age)

            biden_encoding = face_recognition.face_encodings(known_image)[0]
            unknown_encoding = face_recognition.face_encodings(img1)[0]

            results = face_recognition.compare_faces([biden_encoding], unknown_encoding)

            overlay_text = "{}_{}".format(gender, age)
            results_text = "{}".format(results)
            cv2.putText(img1, overlay_text, (x,y), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img1, results_text, (10, 50), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('recognized',img1)
            cnt = cnt + 1