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

print("1")
model=models.load_model("m1测.h5")
dir="C:\\Users\\fyx\\Desktop\\资料下载\\images\\"
for i in range(0,100):
    s=input()
    s=dir+"\\"+s
    img = mpimg.imread(s)
    img=np.asarray([img[:,48:96,:]])
    print(img.shape)
    res=model.predict(img)
    res=res[0]
    num=0
    print(res)
    max=0
    ans=0
    for j in res:
        print(j)
        if max<res[num]:
            max=res[num]
            ans=num
        num+=1
    print(ans)