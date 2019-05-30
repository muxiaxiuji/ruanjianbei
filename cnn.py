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

lena = mpimg.imread('E:\快凉凉的软件杯\imgs\_001a_0.png') # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
lena.shape #(512, 512, 3)
dir="E:\快凉凉的软件杯\imgs"
lis=os.listdir(dir)
i=0
limit=100
for name in lis:
    i+=1
    if i>=limit:
        break
    print(dir+"\\"+name)
    img=mpimg.imread(dir+"\\"+name)
    print(img.shape)
print(lis)
print(lena.shape)
plt.imshow(lena) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()
