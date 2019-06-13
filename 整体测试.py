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
import PictureProcess as pp
import random
import cv2


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
    seq=[]
    s=dir+"\\"+name
    img = cv2.imread(s)
    #img=cv2.resize(img,(46,120))
    for i in range(0,4):
        seq.append(pp.trans(img[:,i*30:i*30+30,:]))
    res=model.predict(np.asarray(seq))
    cnt=0
    for i in range(0,4):
        if str(judge(res[i]))==name[i]:
            cnt+=1
    if cnt==4:
        ac+=1
    else:
        print(str(judge(res[0]))+str(judge(res[1]))+str(judge(res[2]))+str(judge(res[3])),end=" ")
        print(name)
        #pp.show(img)
print(ac/tot)


