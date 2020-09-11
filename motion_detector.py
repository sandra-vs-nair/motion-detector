# -----------------------------------------------------------
# Creating a webcam motion-detector using python.
#
# (C) 2020 Sandra VS Nair, Trivandrum
# email sandravsnair@gmail.com
# -----------------------------------------------------------

import cv2, pandas
from datetime import datetime

#Initializing variables
first_frame=None
status_list=[None,None]
times=[]
df=pandas.DataFrame(columns=["Start","End"])

#The first argument 0 indicates capture video from the first camera available.
video=cv2.VideoCapture(0,cv2.CAP_DSHOW)
while True:
    status=0
    check,frame = video.read()
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  #Convert colored frame to gray.
    gray_frame=cv2.GaussianBlur(gray_frame,(21,21),0)  #Removes noise and increases accuracy for calculation.
    
    #Saving the first frame for comparison. 
    #It should reflect the normal state of the room/building which is being captured.
    if first_frame is None:
        first_frame=gray_frame
        continue
    
    delta_frame=cv2.absdiff(first_frame,gray_frame) #Taking the difference between first frame and the current frame.
    thresh_frame=cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1] #Pixel value 30 is the threshold.
    dilate_frame=cv2.dilate(thresh_frame,None,iterations=2) #For smoothening. Removes little black holes in big white areas.

    (cnts,_)=cv2.findContours(dilate_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)   
    
    #If area of contour greater than 10000 (areas where motion is detected more.), a rectangle is drawn on them.
    for contour in cnts:
        if cv2.contourArea(contour) > 10000:
            status=1
            (x,y,w,h)=cv2.boundingRect(contour)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
    
    status_list.append(status)
    
    #Note down the time when video frames switch from no-motion to motion-detected and viceversa
    if status_list[-1]==1 and status_list[-2]==0 :
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1 :
        times.append(datetime.now())
       
#   cv2.imshow("Gray",gray_frame)
#   cv2.imshow("Delta",delta_frame)
#   cv2.imshow("Threshold",thresh_frame)
#   cv2.imshow("Dilate",dilate_frame)
    cv2.imshow("Capturing",frame) #video Streaming
    
    #Continues streaming till key 'q' is pressed.
    key =cv2.waitKey(1)
    if key==ord('q'):
        if status==1:
            times.append(datetime.now())
        break

for i in range(0,len(times),2):
    df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)

#Saves the noted time list to a csv file.    
df.to_csv("Times.csv")

video.release()
cv2.destroyAllWindows()