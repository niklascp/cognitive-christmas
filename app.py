import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('models/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

INIT_TIME = 100
FRAME_WIDTH = 800
FRAME_HEIGHT = 600

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);

timer = 0;
state = 0

cv2.namedWindow("IMG", cv2.WND_PROP_FULLSCREEN)          
cv2.setWindowProperty("IMG", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

fgbg = cv2.createBackgroundSubtractorMOG2()

background1 = cv2.imread('backgrounds/background1.jpeg', 0)
background1 = cv2.resize(background1 (FRAME_WIDTH, FRAME_HEIGHT))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fgmask = fgbg.apply(frame, learningRate = 0 if state > 0 else -1)

    # Background calibration 
    if state == 0:
        cv2.putText(
        	img = frame,
        	text = 'Init ' + str(INIT_TIME - timer), 
        	org =  (int(FRAME_WIDTH/2 - 100),int(FRAME_HEIGHT/2 - 30)),
        	fontFace = cv2.FONT_HERSHEY_DUPLEX, 
        	fontScale = 2, 
        	color = (255,255,255), 
        	thickness = 5)

        if timer >= INIT_TIME:
        	state = 1
    # Capture    
    if state == 1:  
    
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
        	cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        	roi_gray = gray[y:y+h, x:x+w]
        	roi_color = frame[y:y+h, x:x+w]
        	eyes = eye_cascade.detectMultiScale(roi_gray)
        	for (ex,ey,ew,eh) in eyes:
        		cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


        frame = cv2.bitwise_and(frame, frame, mask=fgmask)

        #cv2.imshow('IMG2', frame_2)
        

    # Display the resulting frame
    cv2.imshow('IMG', frame)
    cv2.imshow('MASK', fgmask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('i'):
    	timer = 0
    	state = 0
    elif key == ord('q'):
        break

    timer = timer + 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()