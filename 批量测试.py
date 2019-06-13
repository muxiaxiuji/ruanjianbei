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
import PictureProcess as pp


def judge(res):
    max=0
    ans=0
    i=0
    for j in res:
        if max<j:
            max=j
            ans=i
        i+=1
    if ans==10:
        return "_"
    else:
        return str(ans)
model=models.load_model("m1测.h5")
dir=r"C:\Users\fyx\Desktop\ruanjianbei\images"
lis=os.listdir(dir)
ac=0
tot=0
for name in lis:
    tot+=1
    img = cv2.imread(dir+"\\"+name)
    rawimg=img[:, 0:30, :]
    img = pp.trans(img[:,0:30,:])
    timg=img
    img=np.asarray([img])
    res=model.predict(img)
    res=res[0]
    #print(res)
    ans=judge(res)
    if str(ans)!=name[0]:
        #print(str(ans))
        """
        plt.imshow(timg)
        plt.show()
        """
        print(name[0]+" "+str(ans))
        #pp.show(rawimg)
        #pp.show(timg)
    else:
        ac+=1
print(ac/tot)


