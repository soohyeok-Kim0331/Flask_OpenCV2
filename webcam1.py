from flask import Flask, g, make_response, Response, request
import cv2

app = Flask(__name__)
video = cv2.VideoCapture(-1)
@app.route('/')
def index():
    return "Default Message!!!"
def gen(video):
    while True:
        success, img = video.read()
        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: img/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_start')
def video_feed():
    global video
    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port='5000')