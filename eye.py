#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[ ]:


face_cascade=cv2.CascadeClassifier(r"C:\Users\hp\open cv\haarcascade_frontalface_default.xml")
eye_cascade=cv2.CascadeClassifier(r"C:\Users\hp\open cv\haarcascade_eye.xml")
smile_cascade=cv2.CascadeClassifier(r"C:\Users\hp\open cv\haarcascade_smile.xml")


# In[ ]:


def detect (gray,frame):
    faces=face_cascade.detectMultiScale(gray,1.5,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h+50,x:x+w+30]
        roi_color=frame[y:y+h+50,x:x+w+30]
        global count
        count+=1
        #cv2.imwrite(r"C:\Users\hp\OneDrive" % count,roi_color)
        eyes=eye_cascade.detectMultiScale(roi_gray,1.1,3)
        smile=smile_cascade.detectMultiScale(roi_gray,1.1,3)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,225),2)
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,225),2)
    return frame
get_face=cv2.VideoCapture(0)
count=0
while True:
    _,frame=get_face.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas=detect(gray,frame)
    cv2.imshow("face detect",canvas)
    if  cv2.waitKey(1) & 0xff==ord("q"):
        break
get_face.release()
cv2.destroyAllWindows()

