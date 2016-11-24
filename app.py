import numpy as np
import cv2

print('Using OpenCV ' + cv2.__version__)

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

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorMOG2(history = INIT_TIME, varThreshold = 12, detectShadows = False)

background1 = cv2.imread('backgrounds/background1.jpeg')
background1 = cv2.resize(background1, (FRAME_WIDTH, FRAME_HEIGHT))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), 0, 0, cv2.INTER_CUBIC);

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fgmask = fgbg.apply(frame, learningRate = 0 if state > 0 else -1)
    #_, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    bgmask = 255 - fgmask

    # Background training 
    if state == 0:
        cv2.putText(
        	img = fgmask,
        	text = 'Training Segmentation Algorithm (' + str(int(100.0*timer/INIT_TIME)) + ' %)', 
        	org =  (0, FRAME_HEIGHT - 5),
        	fontFace = cv2.FONT_HERSHEY_DUPLEX, 
        	fontScale = .7, 
        	color = (255,255,255), 
        	thickness = 1)

        if timer >= INIT_TIME:
        	state = 1

        # Display the resulting frame
        cv2.imshow('IMG', fgmask)

    # Capture    
    if state == 1:

        #m = cv2.moments(fgmask, False);
        #center = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))
        #print(center)
        #center = (int(FRAME_WIDTH/2), int(FRAME_HEIGHT/2))
        #frame = cv2.seamlessClone(frame, background1, fgmask, center, cv2.MIXED_CLONE)
        #frame = cv2.bitwise_and(frame, frame, mask=fgmask)
        #frame = cv2.bitwise_and(background1, background1, mask=bgmask)
        frame = cv2.bitwise_and(background1, background1, mask=bgmask) + cv2.bitwise_and(frame, frame, mask=fgmask)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)        
        for (x,y,w,h) in faces:
           cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
           roi_gray = gray[y:y+h, x:x+w]
           roi_color = frame[y:y+h, x:x+w]
           eyes = eye_cascade.detectMultiScale(roi_gray)
           for (ex,ey,ew,eh) in eyes:
               cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        #cv2.imshow('IMG2', frame_2)
        

        # Display the resulting frame
        cv2.imshow('IMG', frame)

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