#!/usr/bin/python

from flask import Flask, Response, send_from_directory, jsonify
from camera import VideoCamera
import time
import os

app = Flask(__name__)
cam = VideoCamera(usePiCamera=False)
cam.start()

def gen():
    while True:
        time.sleep(1/12)
        frame = cam.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n') 
        

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/images')  
def list_images():
    images = os.listdir('./images')
    return jsonify(*images[-3:][::-1])

@app.route('/images/<path:filename>')  
def send_image(filename):  
    return send_from_directory('images', filename)

@app.route('/<path:filename>')  
def send_file(filename):  
    print(filename)
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, use_reloader=False)
    cam.stop()