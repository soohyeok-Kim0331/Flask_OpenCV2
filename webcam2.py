
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from flask import Flask, g, make_response, Response, request
import cv2

app = Flask(__name__)
video = cv2.VideoCapture(0)

def butter_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if low == 0 and high == 1:
        return data
    elif low == 0 and high != 1:
        b, a = signal.butter(order, highcut / nyq, btype='low')
    elif low != 0 and high == 1:
        b, a = signal.butter(order, lowcut / nyq, btype='high')
    elif low != 0 and high != 1:
        b, a = signal.butter(order, [low, high], btype='band')
    output = signal.filtfilt(b, a, data)
    return output

def measure(directory):
    time = []
    disp = []
    data = []

    frm = 0
    FPS = 30
    param = 1
    sample_len = 100

    green = (0, 225, 0)
    red = (0, 0, 225)

    cap = cv2.VideoCapture(directory)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print(cap.get(3), cap.get(4))

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=30,
                          qualityLevel=0.01,
                          minDistance=5,
                          blockSize=4)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=0,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=100)
    kp, des = orb.detectAndCompute(old_gray, None)

    p0 = cv2.KeyPoint_convert(kp)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    id_dat = []

    while frm <= sample_len:
        ret, frame = cap.read()

        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1
        good_old = p0

        # draw the tracks`
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            id_dat.append([i, b])

        frm = frm + 1

    id_dat = np.array(id_dat)
    id_dat = id_dat.reshape(frm, len(good_new), 2)  # frame, id, elements
    arr = np.zeros((len(good_new), frm))
    for n in range(frm):
        for i in range(len(good_new)):
            arr[i][n] = id_dat[n][i][1]

    std_arr = []
    for i in arr:
        std_arr.append(np.var(i))

    k = 0
    feature_id = 0
    for i in std_arr:
        if i == min(std_arr):
            feature_id = k

        k = k + 1

    while (1):
        ret, frame = cap.read()

        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1
        good_old = p0

        # draw the tracks`
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            if i == feature_id:  # feature id
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a, b), (c, d), green, 2)
                frame = cv2.circle(frame, (a, b), 5, red, -1)

        img = cv2.add(frame, mask)
        k = cv2.waitKey(30) & 0xff

        data.append(b)
        disp = data[0] - data

        disp = disp * param
        time.append(frm / FPS)

        frm = frm + 1

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.imshow('frame', img)

        if k == 27:
            break
        # date the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    return disp, time


@app.route('/')
def index():
    return "Default Message!!!"

@app.route('/video_start')
def video_feed():
    global video
    return Response(measure(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)