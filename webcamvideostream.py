# import the necessary packages
from threading import Thread
import cv2

class WebcamVideoStream:
	def __init__(self, src=0):		
		# initialize the video capure.		
		self.cap = cv2.VideoCapture(src)
		
	def start(self):
		# no op
		pass

	def update(self):
		# no op
		pass

	def read(self):
		(_, frame) = self.cap.read()
		return frame

	def stop(self):
		self.cap.release()