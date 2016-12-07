from videostream import VideoStream
from threading import Thread
import numpy as np
import os
import datetime
import time
import cv2

class VideoCamera(object):
    INIT_TIME = 100
    FRAME_WIDTH = 600
    FRAME_HEIGHT = 480
    HISTORY = 25
    THRESHOLD = 18

    def __init__(self, usePiCamera=True):
        # initialize the camera and grab a reference to the raw camera capture
        self.vs = VideoStream(usePiCamera=usePiCamera, resolution = (self.FRAME_WIDTH, self.FRAME_HEIGHT), framerate = 12)

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history = self.INIT_TIME, varThreshold = 12, detectShadows = False)
        self.face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

        self.background1 = cv2.imread('backgrounds/background1.jpeg')
        self.background1 = cv2.resize(self.background1, (self.FRAME_WIDTH, self.FRAME_HEIGHT))

        self.hats = [cv2.imread('./hats/' + file, cv2.IMREAD_UNCHANGED) for file in os.listdir('./hats')]

        self.stopped = False
        self.frame = None
        self.state = 0
        self.time = 0
        self.lastReady = 0
        self.lastCapture = 0
        self.history = np.zeros(self.HISTORY)

    def start(self):
        self.vs.start()
        # start the thread to process frames from the video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def stop(self):
        self.vs.stop()
        # indicate that the thread should be stopped
        self.stopped = True

    def update(self):
        # Keep looping infinitely until the thread is stopped
        while True:
            # If the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # Grab an image from the video stream
            frame = self.vs.read()
            
            # Wait until frames start to be available
            if frame is None:
                time.sleep(1)
                continue

            # Resize frame to fit working dimensions
            frame = cv2.resize(frame, (self.FRAME_WIDTH, self.FRAME_HEIGHT), 0, 0, cv2.INTER_CUBIC)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            fgmask = self.fgbg.apply(frame, learningRate = 0 if self.state > 0 else -1)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
            bgmask = 255 - fgmask

            # State machine
            # S0: Training 
            if self.state == 0:
                cv2.putText(
                    img = fgmask,
                    text = 'Training Segmentation Algorithm (' + str(int(100.0*self.time/self.INIT_TIME)) + ' %)', 
                    org =  (0, self.FRAME_HEIGHT - 5),
                    fontFace = cv2.FONT_HERSHEY_DUPLEX, 
                    fontScale = .7, 
                    color = (255,255,255), 
                    thickness = 1)

                # Transition to S1
                if self.time >= self.INIT_TIME:
                    self.state = 1
                
                # Show mask when training.
                self.frame = fgmask

            # S1: Ready to capture
            elif self.state == 1:
                # Detect and draw faces
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)    
                for (x,y,w,h) in faces:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                
                # Update history
                self.history[self.time % self.HISTORY] = min(len(faces), 1)

                self.frame = frame
                #self.frame = cv2.bitwise_and(self.background1, self.background1, mask=bgmask) + cv2.bitwise_and(frame, frame, mask=fgmask)

                cv2.putText(
                    self.frame,
                    text = 'Frame ' + str(self.time), 
                    org =  (0, self.FRAME_HEIGHT - 5),
                    fontFace = cv2.FONT_HERSHEY_DUPLEX, 
                    fontScale = .7, 
                    color = (255,255,255), 
                    thickness = 1)

                # Transition to S2 if faces present.
                if np.sum(self.history) >= self.THRESHOLD and self.time > self.lastCapture + self.HISTORY:
                    self.lastReady = self.time
                    self.state = 2
                # Transition to S0 if no faces and time has gone, reset all
                elif np.sum(self.history) < self.THRESHOLD and self.time > 4 * self.INIT_TIME:
                    self.time = 0
                    self.lastReady = 0
                    self.lastCapture = 0
                    self.state = 0
            
            # S2: Cont down
            elif self.state == 2:                
                self.frame = frame

                cv2.putText(
                    self.frame,
                    text = str(3 - int((self.time - self.lastReady) / 10)), 
                    org =  (int(self.FRAME_WIDTH / 2), int(self.FRAME_HEIGHT / 2)),
                    fontFace = cv2.FONT_HERSHEY_DUPLEX, 
                    fontScale = 3, 
                    color = (255,255,255), 
                    thickness = 2)

                if self.time - self.lastReady >= 29:
                    self.state = 3

            # S3: Capture
            elif self.state == 3:
                frame = cv2.bitwise_and(self.background1, self.background1, mask=bgmask) + cv2.bitwise_and(frame, frame, mask=fgmask)

                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5) 

                for (x,y,w,h) in faces:
                    # Select a hat at fit it to face
                    hat = self.hats[0]
                    hat = cv2.resize(hat, (w, int(w/hat.shape[1]*hat.shape[0])), 0, 0, cv2.INTER_CUBIC)
                    # Get fitted hat width/hight, and crop to bounding box.
                    h_h, h_w = hat.shape[:2] 
                    h_x1 = x
                    h_x2 = min(x + w, self.FRAME_WIDTH)
                    h_y1 = max(y - h_h, 0)
                    h_y2 = min(y, self.FRAME_HEIGHT)
                    # Recalulate hat width/hight if cropped by bounding box.
                    h_w = h_x2 - h_x1
                    h_h = h_y2 - h_y1

                    # Blend hat to frame with alpha-channel.
                    for c in range(0,3):
                        alpha = hat[-h_h:, :h_w, 3] / 255.0
                        color = hat[-h_h:, :h_w, c] * alpha
                        beta  = frame[h_y1:h_y2, h_x1:h_x2, c] * (1.0 - alpha)
                        frame[h_y1:h_y2, h_x1:h_x2, c] = color + beta

                # Save the image to disk.
                filename = 'images/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png';
                print('Saving image: ', filename)
                cv2.imwrite(filename, frame)

                self.frame = frame                
                time.sleep(10)

                # Transition to S1
                self.lastCapture = self.time
                self.state = 1

            self.time = self.time + 1
            time.sleep(1/16)

    def get_frame(self):
        # Wait until frames start to be available
        while self.frame is None:
            time.sleep(0)

        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', self.frame)
        return jpeg.tobytes()