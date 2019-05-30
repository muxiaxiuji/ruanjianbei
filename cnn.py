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

def creatmodel():
    mo = models.Sequential()
    mo.add(layers.Conv2D(filters=15,strides=[4,4], kernel_size=[9, 9], activation='relu', padding='same', input_shape=[46, 120,3]))
    mo.add(layers.MaxPooling2D(pool_size=(2,2),strides=[2,2],padding='same'))
    mo.add(layers.Conv2D(filters=30,strides=[2,2], kernel_size=[4, 4], activation='relu', padding='same', input_shape=[46, 120,3]))
    mo.add(layers.MaxPooling2D(pool_size=(2,2),strides=[2,2],padding='same'))
    mo.add(layers.Flatten())
    mo.add(layers.Dense(units=80, activation='relu',trainable='true'))
    mo.add(layers.Dense(units=40, activation='relu',trainable='true'))
    mo.add(layers.Dense(units=20, activation='relu', trainable='true'))
    mo.add(layers.Dense(units=11,activation="softmax" ))

    #mo.add(layers.Conv2D(filters=19,kernel_size=[1,1], activation='relu'))
    #mo.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    mo.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(mo.summary())
    return  mo;

def vectorize_sequences(sequences, dimension=11):
    results = np.zeros((len(sequences), dimension))
    print (results.shape)
    for i, sequence in enumerate(sequences):
        print (i)
        print (sequence)
        if sequence=="_":
            num=10
        else:
            num=int(sequence)
        #print (type(sequence))
        results[i, num] = 1.
    return results

model=creatmodel()
lena = mpimg.imread('E:\快凉凉的软件杯\imgs\_001a_0.png') # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
lena.shape #(512, 512, 3)
dir="E:\快凉凉的软件杯\imgs"
lis=os.listdir(dir)
random.shuffle(lis)
i=-1
limit=1000
a=[]
sequence=""
for name in lis:
    #print(name[1])
    #print(sequence)
    i+=1
    if i>=limit:
        break
    #print(dir+"\\"+name)
    sequence = sequence + name[1]
    img=mpimg.imread(dir+"\\"+name)
    a.append(img)
    #print(img.shape)
print(sequence)
tags=vectorize_sequences(sequence, dimension=11)
a=np.asarray(a)
print(a.shape)
#print(lis)
print(lena.shape)
x_val = a[:100]
partial_x_train = a[100:]
y_val = tags[:100]
partial_y_train = tags[100:]
history = model.fit(partial_x_train, partial_y_train, epochs=300,batch_size=512,validation_data=(x_val, y_val))
#model.predict()
#history = model.fit(partial_x_train, partial_y_train, epochs=300,batch_size=512,validation_data=(x_val, y_val))
#model.save("m测.h5")
#plt.imshow(lena) # 显示图片
#plt.axis('off') # 不显示坐标轴
#plt.show()

import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

results = model.evaluate(x_val, y_val)
print (results)

model.save("m1测.h5")