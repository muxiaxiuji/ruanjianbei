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

def creatmodel():
    mo = models.Sequential()
    mo.add(layers.Conv2D(filters=3, kernel_size=[9, 9], activation='relu', padding='same', input_shape=[46, 120,3]))
    #mo.add(layers.MaxPooling2D(pool_size=(2,2),padding='same'))
    mo.add(layers.Dense(units=64*64, activation='relu',trainable='true'))
    mo.add(layers.Dense(units=32*32, activation='relu',trainable='true'))
    mo.add(layers.Dense(units=19*19, activation='relu',trainable='true'))
    mo.add(layers.Dense(units=4, activation='relu',trainable='true'))
    #mo.add(layers.Conv2D(filters=19,kernel_size=[1,1], activation='relu'))
    mo.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    print(mo.summary())
    return  mo;

model=creatmodel()
lena = mpimg.imread('E:\快凉凉的软件杯\imgs\_001a_0.png') # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
lena.shape #(512, 512, 3)
dir="E:\快凉凉的软件杯\imgs"
lis=os.listdir(dir)
i=-1
limit=100
a=[]
for name in lis:
    print(name)
    i+=1
    if i>=limit:
        break
    print(dir+"\\"+name)
    img=mpimg.imread(dir+"\\"+name)
    a.append(img)
    print(img.shape)
a=np.asarray(a)
print(a.shape)
#print(lis)
print(lena.shape)

#plt.imshow(lena) # 显示图片
#plt.axis('off') # 不显示坐标轴
#plt.show()
