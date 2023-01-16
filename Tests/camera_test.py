import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import Transform
import time

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480),
                                                         # Yes, OpenCV wants BGR. Somehow telling picamera
                                                         # this gets the right colors. Go figure....
                                                         "format": "XRGB8888"},
                                                   transform=Transform(hflip=True, vflip=True),
                                                   buffer_count=2))
picam2.start()

start = 0
n_frame = 0
show_start = 100
try:
    while True:
        image = picam2.capture_array()
        n_frame += 1
        if n_frame % 10 == 0:
            show = cv2.resize(image, (160, 120))
            cv2.imshow("Image", show)
            cv2.waitKey(1)      # Force OpenCV to actually show something.
        if n_frame == show_start:
            # assume that the first few calls to imshow do a lot of setup.
            start = time.perf_counter()

except KeyboardInterrupt:
    print("Quitting")

elapsed = time.perf_counter() - start
print((n_frame - show_start) / elapsed, "frames per second")
cv2.destroyAllWindows()