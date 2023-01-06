import time # Provides time-related functions
start = time.perf_counter_ns()

# tty.setcbreak(sys.stdin)

# import the necessary packages
from picamera.array import PiRGBArray # Generates a 3D RGB array
from picamera import PiCamera # Provides a Python interface for the RPi Camera Module
import cv2 # OpenCV library
import numpy as np # Numpy Library

print("Starting HSV Select Live")

window_name = 'HSV Select Live'

cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

hsv_min = np.array([0, 0, 0])
hsv_max = np.array([179, 255, 255])

resolution = (640, 480)

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def update():

    try:

        with PiCamera() as camera:

            camera.resolution = resolution

            camera.vflip = True

            camera.framerate = 30

            raw_capture = PiRGBArray(camera, size=resolution)

            for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):

                image = frame.array

                hsv_min[0] = cv2.getTrackbarPos('H_min', window_name)
                hsv_min[1] = cv2.getTrackbarPos('S_min', window_name)
                hsv_min[2] = cv2.getTrackbarPos('V_min', window_name)

                hsv_max[0] = cv2.getTrackbarPos('H_max', window_name)
                hsv_max[1] = cv2.getTrackbarPos('S_max', window_name)
                hsv_max[2] = cv2.getTrackbarPos('V_max', window_name)

                image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                mask = cv2.inRange(image_hsv, hsv_min, hsv_max)

                mask = cv2.bitwise_and(image, image, mask = mask)

                h, w, ch = image.shape

                cv2.putText(mask, 'HSV Lower: {}'.format(hsv_min), (10, 35), 0, 0.75, (255, 0, 0), 2)

                cv2.putText(mask, 'HSV Upper: {}'.format(hsv_max), (10, 70), 0, 0.75, (255, 0, 0), 2)

                resized_mask = ResizeWithAspectRatio(mask, width=300)

                cv2.imshow(window_name, mask)

                key = cv2.waitKey(1) & 0xFF

                raw_capture.truncate(0)

                if key == ord("q"):
                    break

                time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        return


def callback(value):
    update()


# make the trackbar used for HSV masking
cv2.createTrackbar('H_min', window_name, 0, 179, callback)
cv2.createTrackbar('S_min', window_name, 0, 255, callback)
cv2.createTrackbar('V_min', window_name, 0, 255, callback)

cv2.createTrackbar('H_max', window_name, 0, 179, callback)
cv2.createTrackbar('S_max', window_name, 0, 255, callback)
cv2.createTrackbar('V_max', window_name, 0, 255, callback)


# set initial trackbar values
cv2.setTrackbarPos('H_min', window_name, 0)
cv2.setTrackbarPos('S_min', window_name, 0)
cv2.setTrackbarPos('V_min', window_name, 0)

cv2.setTrackbarPos('H_max', window_name, 0)
cv2.setTrackbarPos('S_max', window_name, 0)
cv2.setTrackbarPos('V_max', window_name, 0)


# wait for 'ESC' destroy windows
while cv2.waitKey(200) & 0xFF != 27:
    update()


cv2.destroyAllWindows()
cv2.waitKey(1)