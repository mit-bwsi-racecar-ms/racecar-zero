from picamera2 import Picamera2 # Provides a Python interface for the RPi Camera Module
from libcamera import Transform

import time # Python time functions
import cv2 # OpenCV computer vision library
import numpy as np # NumPy library for numerical programming in Python

print("Starting HSV Select Live")

window_name = 'HSV Select Live'

cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

hsv_min = np.array([0, 0, 0])
hsv_max = np.array([179, 255, 255])

# resolution = (640, 480)
resolution = (320, 240)
# resolution = (960, 720)

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

    nFrames = 0
    show_start = 100

    try:
        while True:

            image = picam2.capture_array()

            nFrames += 1

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

            show = cv2.resize(image, (160, 120))
            cv2.imshow(window_name, mask)
            cv2.waitKey(1)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            time.sleep(0.1)

            if nFrames == show_start:
                # assume that the first few calls to imshow do a lot of setup.
                start = time.perf_counter()

    except KeyboardInterrupt:
        pass
    finally:
        return

    elapsed = time.perf_counter() - start
    print((nFrames - show_start) / elapsed, "frames per second")
    cv2.destroyAllWindows()

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
cv2.setTrackbarPos('H_max', window_name, 179)
cv2.setTrackbarPos('S_max', window_name, 255)
cv2.setTrackbarPos('V_max', window_name, 255)

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480),
                                                        # Yes, OpenCV wants BGR. Somehow telling picamera
                                                        # this gets the right colors. Go figure....
                                                        "format": "XRGB8888"},
                                                transform=Transform(hflip=True, vflip=True),
                                                buffer_count=2))
picam2.start()

# wait for 'ESC' destroy windows
while cv2.waitKey(200) & 0xFF != 27:
    update()


cv2.destroyAllWindows()
cv2.waitKey(1)