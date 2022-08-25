import cv2
import pickle
import numpy as np
import face_recognition

cap=cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier('D:\\Data\\work\\haarcascade_frontalface_default.xml')
classifier_face = cv2.CascadeClassifier('D:\\Data\\work\\haarcascade_eye.xml')

pickle_in = open('D:\\Data\\work\\model.sav','rb')
loaded_model = pickle.load(pickle_in)
pickle_in.close()

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_copy = frame.copy()
    face = face_recognition.face_locations(gray)
    face_loc = list(face)
    print
    
    #detect = classifier.detectMultiScale(gray,1.2,1)
    #cv2.rectangle(frame,(face_loc[3],face_loc[0]),(face_loc[1],face_loc[2]),(0,255,0),2)
        # roi_gray = gray[y:y+h,x:x+w]
        # roi_color = frame[y:y+h,x:x+w]
        # eyes =classifier_face.detectMultiScale(roi_gray,1.2,1)
        # for (ex, ey, ew, eh) in eyes:
        #     roi_gray.reshape(80,80)
        #     cv2.rectangle(roi_color,(ex, ey), (ex+ew,ey+eh),(255,0,0),2)
        #     # print('The roi_gray is :{}'.format(np.array(roi_gray)))
        #     # print('The frame is :{}'.format(np.array(frame)))
        #     eye_state = np.array(roi_gray).flatten()
        #     # print(eye_state.size)
        #     # print(eye_state.shape)
        #     # print(eye_state.ndim)
            

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xff==ord('q'):
        break
cap.release()  
cv2.destroyAllWindows