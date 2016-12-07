#!/usr/bin/python

from flask import Flask, Response, send_from_directory
from camera import VideoCamera
import time

app = Flask(__name__)
cam = VideoCamera(usePiCamera=False)
cam.start()

def gen():
    while True:
        frame = cam.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n') 
        time.sleep(1.0/12)      

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/<path:filename>')  
def send_file(filename):  
    print(filename)
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, use_reloader=False)
    cam.stop()