import keras
import numpy as np
import  gensim
import time
from keras import models
from keras import layers
from keras import regularizers
import os
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import random
import cv2

size = [46,30,1]
def minus(a,b):
    if a>b:
        return a-b
    else:
        return b-a
def isvalid(x, y):
    return 0 <= x < size[0] and 0 <= y < size[1]

def trans(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    """
    ret, th1 = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
    ans = th2
    ans=cv2.erode(ans,np.ones((3,3),))
    """
    newimg = np.zeros([46, 30, 1])
    for i in range(0, 46):
        for j in range(0, 30):
            tmp = 0
            step = 1
            for x in range(i-step,i+step+1):
                for y in range(j-step,j+step+1):
                    if isvalid(x,y):
                        tmp=max(tmp,minus(img[i,j],img[x,y]))
            newimg[i, j, 0] = tmp
    return newimg

def show(img):
    cv2.namedWindow("Image")
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def judge(res):
    max=0
    ans=0
    i=0
    for j in res:
        if max<j:
            max=j
            ans=i
        i+=1
    return ans
model=models.load_model("m1测.h5")
dir=r"C:\Users\fyx\Desktop\ruanjianbei\images"
lis=os.listdir(dir)
ac=0
tot=0
for name in lis:
    tot+=1
    seq=[]
    s=dir+"\\"+name
    img = cv2.imread(s)
    for i in range(0,4):
        seq.append(trans(img[:,i*30:i*30+30,:]))
    res=model.predict(np.asarray(seq))
    cnt=0
    for i in range(0,4):
        if str(judge(res[i]))==name[i] or name[i]=='_':
            cnt+=1
    if cnt==4:
        ac+=1
print(ac/tot)


