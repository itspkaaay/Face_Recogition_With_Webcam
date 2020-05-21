import numpy as np 
import cv2
import os

############### KNN Code #################
def dist(X,Y):
    return np.sqrt(np.sum((X-Y)**2))

def knn(Points_arr,Class_Arr,Query_point,k=5):
    Values=[] #is a tuple containing distance and class of a given point wrt to query point
    for i in range(Class_Arr.shape[0]):
        d= dist(Query_point,Points_arr[i])
        Values.append((d,Class_Arr[i]))
    
    Values= sorted(Values) # sorts the tuple on the basis of the first parameter
    Values= Values[:k]
    Values= np.array(list(Values)) #convert list of tuples to numpy list of list for numpy opertaion unique and return count
    print(Values)
    new_Vals= np.unique(Values[:,1],return_counts=True)
    index_max= new_Vals[1].argmax()
    return new_Vals[0][index_max]

##### Training Data Parameters #################

datasetPath= '/Users/pk/Desktop/Python for DS course/face_recognition_test_data/'
class_id= 0
data= []     #contains the input data
label= []    #contains the corresponding output/ class_id
names= {                                                #Dictionary mapping class_id and name of 
    
}

###### Data preparation- Training Data #####################

for fx in os.listdir(datasetPath):
    if fx.endswith('.npy'):
        names[class_id]= fx[0:-4]
        data_item= np.load(datasetPath+fx)
        print(fx,type(data_item),data_item.shape)
        data.append(data_item)
        target= class_id * np.ones((data_item.shape[0],))
        class_id+=1
        label.append(target)
        
data= np.concatenate(data,axis=0)
label= np.concatenate(label,axis=0)
trainingdataset= np.concatenate((data,label.reshape((-1,1))),axis=1)
print(data.shape)
print(label.shape)
print(trainingdataset.shape)
print(trainingdataset[:,-1])


########## Testing the Classifier ############

cap1= cv2.VideoCapture(0)
face_detect= cv2.CascadeClassifier('/Users/pk/Desktop/Python for DS course/haarcascade_frontalface_alt.xml')
while(True):
    ret, frame= cap1.read()
    if ret==False:
        continue
    frame_cropped=frame
    faces_test= face_detect.detectMultiScale(frame,1.3,5)
    
    for f in faces_test:
        (x,y,h,w)=f
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        offset=10
        frame_cropped= frame[y-offset:y+h+offset,x-offset:x+w+offset]
        frame_cropped= cv2.cvtColor(frame_cropped,cv2.COLOR_BGR2GRAY)
        face_to_test=cv2.resize(frame_cropped,(100,100))
        out= knn(trainingdataset[:,:10000],trainingdataset[:,10000],face_to_test.flatten(),5)
        #print(out)
        pred_name= names[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    cv2.imshow('faces',frame)
    
    key_pressed= cv2.waitKey(1) & 0xFF
    if key_pressed== ord('q'):
        break

print(names.keys(),names.values())
cap1.release()
cv2.destroyAllWindows()
