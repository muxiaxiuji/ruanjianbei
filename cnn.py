import keras
import numpy as np
import gensim
import time
import cv2
from keras import models
from keras import layers
from keras import regularizers
import os
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np
import random

size = [46, 30, 1]


def creatmodel():
    mo = models.Sequential()
    mo.add(layers.Conv2D(filters=24, strides=[1, 1], kernel_size=[5, 5], activation='relu', padding='same',
                         input_shape=size))
    mo.add(layers.MaxPooling2D(pool_size=(2, 2), strides=[2, 2], padding='same'))
    mo.add(layers.Conv2D(filters=48, strides=[1, 1], kernel_size=[5, 5], activation='relu', padding='same'))
    mo.add(layers.MaxPooling2D(pool_size=(2, 2), strides=[2, 2], padding='same'))
    mo.add(layers.Flatten())
    mo.add(layers.Dense(units=256, activation='relu', trainable='true'))
    mo.add(layers.Dropout(0.5))
    mo.add(layers.Dense(units=64, activation='relu', trainable='true'))
    mo.add(layers.Dropout(0.5))
    mo.add(layers.Dense(units=10, activation="softmax"))

    # mo.add(layers.Conv2D(filters=19,kernel_size=[1,1], activation='relu'))
    # mo.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    mo.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(mo.summary())
    return mo


def vectorize_sequences(sequences, dimension=10):
    results = np.zeros((len(sequences), dimension))
    print(results.shape)
    for i, sequence in enumerate(sequences):
        print(i)
        print(sequence)
        if sequence == "_":
            num = 10
        else:
            num = int(sequence)
        # print (type(sequence))
        results[i, num] = 1.
    return results


def minus(a, b):
    if a > b:
        return a - b
    else:
        return b - a


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
            for x in range(i - step, i + step + 1):
                for y in range(j - step, j + step + 1):
                    if isvalid(x, y):
                        tmp = max(tmp, minus(img[i, j], img[x, y]))
            newimg[i, j, 0] = tmp
    return newimg


def show(img):
    cv2.namedWindow("Image")
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


model = creatmodel()
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
dir = r"C:\Users\fyx\Desktop\ruanjianbei\images"
lis = os.listdir(dir)
random.shuffle(lis)
limit = 1000
a = []
sequence = ""
for i in range(0, limit - 1):
    name = lis[i]
    wh = random.randint(0, 3)
    while name[wh] == "_":
        wh = random.randint(0, 3)
    sequence = sequence + name[wh]
    img = cv2.imread(dir + '\\' + name)
    img = trans(img[:, wh * 30:(wh + 1) * 30, :])
    # show(img)
    a.append(img)
print(sequence)
tags = vectorize_sequences(sequence, dimension=10)
a = np.asarray(a)
print(a.shape)
x_val = a[:100]
partial_x_train = a[100:]
y_val = tags[:100]
partial_y_train = tags[100:]
history = model.fit(partial_x_train, partial_y_train, epochs=150, batch_size=300, validation_data=(x_val, y_val))
# model.predict()
# history = model.fit(partial_x_train, partial_y_train, epochs=300,batch_size=512,validation_data=(x_val, y_val))
# model.save("m测.h5")
# plt.imshow(lena) # 显示图片
# plt.axis('off') # 不显示坐标轴
# plt.show()

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
print(results)

model.save("m1测.h5")
