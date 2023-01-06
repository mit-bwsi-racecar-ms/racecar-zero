import RPi.GPIO as GPIO # Library for handling GPIO pins
from rpi_hardware_pwm import HardwarePWM # Hardware PWM functions

from picamera import PiCamera # Provides a Python interface for the RPi Camera Module
from picamera.array import PiRGBArray # Generates a 3D RGB array

import time # Python time functions
import cv2 # OpenCV computer vision library
import numpy as np # NumPy library for numerical programming in Python
import sys # Some system functions, in case we need them for debugging


### Motor functions and setup

# Set the way pin numbers are associated with actual pins
GPIO.setmode(GPIO.BCM)

# GPIO pin assignments
rightMotor1 = 24
rightMotor2 = 23
# pwm 0 = 18
leftMotor1 =  6
leftMotor2 = 13
# pwm 1 = 19
standby =  5
quitStartup = 7

# Right motors
GPIO.setup(rightMotor1,GPIO.OUT)
GPIO.setup(rightMotor2,GPIO.OUT)
GPIO.output(rightMotor1,GPIO.LOW)
GPIO.output(rightMotor2,GPIO.LOW)
# Left motors
GPIO.setup(leftMotor1,GPIO.OUT)
GPIO.setup(leftMotor2,GPIO.OUT)
GPIO.output(leftMotor1,GPIO.LOW)
GPIO.output(leftMotor2,GPIO.LOW)
# Standby
GPIO.setup(standby, GPIO.OUT)
GPIO.output(standby, GPIO.LOW)
GPIO.setup(quitStartup, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# The maximum frequency as specified in the datasheet for the L298 and TB6612FNG
# motor driver we are using
MAX_FREQUENCY_L298 = 25000
MAX_FREQUENCY_TB6612FNG = 100000
FREQUENCY_PWM = MAX_FREQUENCY_TB6612FNG
pwmRight = HardwarePWM(0, hz=FREQUENCY_PWM)
pwmLeft  = HardwarePWM(1, hz=FREQUENCY_PWM)
pwmLeft.start(0)    # 0 duty cycle = motor off
pwmRight.start(0)

def setLeftForward():
    GPIO.output(leftMotor1,GPIO.HIGH)
    GPIO.output(leftMotor2,GPIO.LOW)
def setLeftBackward():
    GPIO.output(leftMotor1,GPIO.LOW)
    GPIO.output(leftMotor2,GPIO.HIGH)
def setRightForward():
    GPIO.output(rightMotor1,GPIO.HIGH)
    GPIO.output(rightMotor2,GPIO.LOW)
def setRightBackward():
    GPIO.output(rightMotor1,GPIO.LOW)
    GPIO.output(rightMotor2,GPIO.HIGH)
def setGPIOForward():
    setLeftForward()
    setRightForward()
def setGPIOBackward():
    setLeftBackward()
    setRightBackward()
def setGPIOClockwise():
    setLeftForward()
    setRightBackward()
def setGPIOCounterClockwise():
    setLeftBackward()
    setRightForward()

def stop():
    GPIO.output(standby, GPIO.LOW)
    pwmLeft.stop()
    pwmRight.stop()

def cleanup():
    GPIO.cleanup()

def clamp(x, minValue, maxValue):
    return min(maxValue, max(minValue, x))

### Do a little dance so we know this program has started

danceStepSeconds = .3
danceSpeed = 40                 # 40% speed

pwmLeft.change_duty_cycle(danceSpeed)
pwmRight.change_duty_cycle(danceSpeed)

setGPIOClockwise()
GPIO.output(standby, GPIO.HIGH) # Enable the motors
time.sleep(danceStepSeconds)
GPIO.output(standby, GPIO.LOW) # Stop the motors
time.sleep(danceStepSeconds)
setGPIOCounterClockwise()
GPIO.output(standby, GPIO.HIGH) # Enable the motors
time.sleep(danceStepSeconds)
GPIO.output(standby, GPIO.LOW) # Stop the motors
pwmLeft.change_duty_cycle(0)
pwmRight.change_duty_cycle(0)
time.sleep(1)


### Camera functions and setup

# Choose an image resolution that gives us enough pixels to see the cone at a
# reasonable distance, but is small enough that we can process fast enough.
resolution = (320, 240)

# HSV values for orange
# TODO: Change the values in the HSV lower bound and upper bound based on the HSV select for your cone
ORANGE_HSV_LOWER_BOUND = np.array([0, 0, 0])
ORANGE_HSV_UPPER_BOUND = np.array([179, 255, 255])

NORMAL_DRIVE_SPEED = 50

# TODO: Change these parameters to see what works best for your cone
IGNORABLE_BLOB_SIZE = 1000
TOO_SMALL_SIZE = 8000
TOO_BIG_SIZE = 15000

# Use a simple IIR Filter for averaging the size.
blobSizeFiltered = 0
count = 0

# Here's the main function.
# Note that it is not called until the end of this file.
def followCone():
    try:

        nFrames = 0

        # Initialize the camera
        with PiCamera() as camera:

            camera.resolution = resolution

            # The camera is mounted upside down, so flip the image vertically
            camera.vflip = True

            # Wait a tenth of a second to allow the camera time to warmup
            time.sleep(0.1)

            # Generates a 3D RGB array and stores it in rawCapture
            rawCapture = PiRGBArray(camera, size=resolution)

            GPIO.output(standby, GPIO.HIGH)

            print("Start")
            sys.stdout.flush()

            # Capture frames continuously from the camera
            for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

                nFrames += 1

                print("here")

                # Grab the raw NumPy array representing the image
                image = frame.array

                orangeBlobs = findBlobs(image, ORANGE_HSV_LOWER_BOUND, ORANGE_HSV_UPPER_BOUND)

                if len(orangeBlobs) > 0:
                    largestBlob = max(orangeBlobs, key = cv2.contourArea)
                    blobSize = cv2.contourArea(largestBlob)
                    global blobSizeFiltered
                    # Do a little averaging of the size, since individual measurements are noisy
                    blobSizeFiltered = (0.6 * blobSize) + (0.4 * blobSizeFiltered)

                    # Print the size of the contour for debugging
                    print(blobSizeFiltered)

                    # Check if the contour/object should or should not be ignored
                    if blobSizeFiltered > IGNORABLE_BLOB_SIZE:
                        # Call the getSpeed and getDirection function to determine how to drive the car
                        drive(getSpeed(blobSizeFiltered), getDirection(largestBlob))
                    else:
                        drive(0, 0)

                    # For debugging, displays the frames that the camera gets

                    # if nFrames % 10 == 0:
                    #     cv2.drawContours(image, [largestBlob], -1, (0, 255, 0), 3)
                    #     cv2.imshow("Frame", image)

                # Done with that frame, clear it
                rawCapture.truncate(0)

    except KeyboardInterrupt:
        # If the user types control-C, stop
        pass
    finally:
        print("Stop")
        stop()
        cleanup()



def findBlobs(image, HSVLowerBound, HSVUpperBound):
    # Convert the image to the HSV color space
    imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Find the pixels in the specifed color range
    colorPixels = cv2.inRange(imageHSV, HSVLowerBound, HSVUpperBound)

    # In computer vision, a "contour" is the boundary of a group of like pixels (a "blob")
    # Return a list of the boundaries of the blobs
    return cv2.findContours(colorPixels, 3, 2)[0]

screenWidth = resolution[0]
screenHeight = resolution[1]
screenCenter = screenWidth/2

# Use a simple IIR Filter for averaging the size and position.
centerFiltered = 0

def getSpeed(blobSize):
    # If the blob is too small, move forward
    if blobSize <= TOO_SMALL_SIZE:
        return  NORMAL_DRIVE_SPEED
    # If the blob is too big, move backward
    if blobSize >= TOO_BIG_SIZE:
        return -NORMAL_DRIVE_SPEED
    # Otherwise, stay still
    return 0

# The fraction of the screen we are away from the center
# Positive if the target is to the right of center
def getDirection(blob):

    # find center of blob (detected cone)
    M = cv2.moments(blob)
    if M['m00'] != 0.0:

        blobCenter = int(M['m10']/M['m00'])

        global centerFiltered
        # Filter (average) the center
        centerFiltered = (0.8 * blobCenter) + (0.2 * centerFiltered)

        return (centerFiltered - float(screenCenter)) / screenWidth

    return 0


# Positive speed is forward
# Positive direction is right, negative is left
def drive(speed, direction):
    if speed < 0:
        setGPIOBackward()
    else:
        setGPIOForward()

    absSpeed = abs(speed)

    pwmLeft.change_duty_cycle(clamp(absSpeed - direction * absSpeed, 0, 100))
    pwmRight.change_duty_cycle(clamp(absSpeed + direction * absSpeed, 0, 100))


# Run the program!
followCone()