import cv2
import numpy as np
import os
import random
import PictureProcess as pp

filenum = 1000
filedir = r"C:\Users\fyx\Desktop\ruanjianbei\test_images"


def randfile():
    flis = os.listdir(filedir)
    random.shuffle(flis)
    return flis[:filenum]


lis = randfile()
for name in lis:
#name = lis[0]
    img = cv2.imread(filedir+'\\'+name)
    img = pp.trans(img)
    img = pp.threshold(img)
    pp.show(img)