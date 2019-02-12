import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
from random import randint


lower_blue = np.array([115,50,50])
upper_blue = np.array([130,255,255])
# for idx in range(1,22):
im = cv2.imread("3157"+".png")
im = imutils.resize(im, height=300)
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower_blue, upper_blue)
im = imutils.resize(mask, height=10)
_,contours,_ = cv2.findContours(im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print(len(contours)) 
cv2.imshow("im",im)
cv2.waitKey(0)
		


# # upper mask (175-180)
# lower_red = np.array([175,110,110])
# upper_red = np.array([180,255,255])
# mask = cv2.inRange(img_hsv, lower_red, upper_red)
# # upper mask (170-180)
# lower_red_2 = np.array([0,120,120])
# upper_red_2 = np.array([5,255,255])
# mask2 = cv2.inRange(img_hsv, lower_red_2, upper_red_2)

# # join my masks
# mask = mask + mask2




# pic[np.where(mask==0)] = 0

