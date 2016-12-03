from imutils.video import VideoStream
import time
import cv2

class VideoCamera(object):
    def __init__(self, usePiCamera=True):
        # initialize the camera and grab a reference to the raw camera capture
        self.vs = VideoStream(usePiCamera=usePiCamera).start()
        time.sleep(2.0)

    def __del__(self):
        self.vs.stop()
    
    def get_frame(self):
        # grab an image from the camera
        frame = self.vs.read()

        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()