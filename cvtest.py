import cv2
from imutils import contours
import numpy as np
import argparse
import imutils
from matplotlib import pyplot as plt

dir = "C:\\Users\\fyx\\Desktop\\ruanjianbei\\images\\"
img = cv2.imread(r"C:\Users\fyx\Desktop\ruanjianbei\test_images\9.jpeg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret,th1=cv2.threshold(img,70,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,5)

cv2.namedWindow("Image")
cv2.imshow("Image", th1)
cv2.waitKey (0)
cv2.destroyAllWindows()