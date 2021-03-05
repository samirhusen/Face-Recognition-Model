import cv2
import numpy as np
import face_recognition

imgSam = face_recognition.load_image_file('ImagesBasic/Samir.jpg')
imgSam = cv2.cvtColor(imgSam,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/BirajTest.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

#face location capture wher the face is
faceLoc = face_recognition.face_locations(imgSam)[0]
encodeSam = face_recognition.face_encodings(imgSam)[0]
cv2.rectangle(imgSam,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLoc = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

#SVM classifier
results = face_recognition.compare_faces([encodeSam],encodeTest)
faceDis = face_recognition.face_distance([encodeSam],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Samir',imgSam)
cv2.imshow('SamirTest',imgTest)
cv2.waitKey(0)
