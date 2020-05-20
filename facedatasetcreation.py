import cv2
import numpy as np 
filepath='/Users/pk/Desktop/Python for DS course/'
face_data=[]
cap= cv2.VideoCapture(0)
name= input('input name of person')
ctr=1
face_detect= cv2.CascadeClassifier('/Users/pk/Desktop/Python for DS course/haarcascade_frontalface_alt.xml')
while True:
    ret, frame= cap.read()
    if ret==False:
        continue
    
    frame_gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame_gray_cropped=frame_gray
    faces=face_detect.detectMultiScale(frame,1.3,5)
    offset=10
    for face in faces:
        (x,y,w,h)=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    frame_gray_cropped= frame_gray[y-offset:y+h+offset,x-offset:x+w+offset]
    face_to_save=cv2.resize(frame_gray_cropped,(100,100))
    ctr=ctr+1
    if ctr%10==0:                        #Storing every 10th frame from webcam video Stream
        face_data.append(face_to_save)
        print(len(face_data)," saved")
    cv2.imshow('frame rgb',frame)
    cv2.imshow('frame cropped gray',frame_gray_cropped)
    
    key_pressed= cv2.waitKey(1) & 0xFF
    if key_pressed== ord('q'):
        break

#converty face list into an array to a numpy array
face_data= np.asarray(face_data)
face_data= face_data.reshape(face_data.shape[0],-1)
print(face_data.shape)

np.save(filepath+name,".npy")
print('Data Collected')

cap.release()
cv2.destroyAllWindows()
    
