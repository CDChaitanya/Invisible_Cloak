# INVISIBLE CLOAK

import cv2
import numpy as np
import time

cam = cv2.VideoCapture(0)
time.sleep(3)
background = 0
for i in range(30):
    ret , background = cam.read()

background = np.flip(background , axis = 1)

while(cam.isOpened()):
    ret , image = cam.read()
    # FLIPPING
    image = np.flip(image , axis=1)
    # CONVERTING IMAGE TO HSV
    hsv = cv2.cvtColor(image , cv2.COLOR_BGR2HSV)
    # USING GAUSSIAN BLUR
    blur = cv2.GaussianBlur(src=hsv, ksize=(35,35), sigmaX=0)
    # DEFINING LOWER RANGE FOR RED COLOR DETECTION
    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])
    mask_1 = cv2.inRange(src=hsv, lowerb=lower_red, upperb=upper_red)
    # DEFINING LOWER RANGE FOR RED COLOR DETECTION
    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask_2 = cv2.inRange(src=hsv, lowerb=lower_red, upperb=upper_red)
    # ADDING BOTH THE MASK TO GENRATE FINAL MASK
    mask = mask_1 + mask_2
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    # REPLACING PIXELS CORRESPONDING TO CLOAK WITH THE BACKGROUND PIXELS
    image[np.where(mask==255)] = background[np.where(mask==255)]
    cv2.imshow('Invisible Cloak' , image)
    if cv2.waitKey(1) == 13:
        break

cam.release()
cv2.destroyAllWindows()

