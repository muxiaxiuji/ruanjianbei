import keras
import numpy as np
import cv2
from keras.models import *
from keras.layers import *

from keras import regularizers
import os
import PictureProcess as pp
import matplotlib.pyplot as plt  # plt 用于显示图片
import numpy as np
import random

size = [46, 30, 1]

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def creatmodel():
    """
    mo = models.Sequential()
    mo.add(layers.Input(shape=(46,None,1)))
    mo.add(layers.Conv2D(filters=32, strides=[1, 1], kernel_size=[3, 3], padding='same'))
    mo.add(layers.BatchNormalization())
    mo.add(layers.Activation("relu"))
    mo.add(layers.Conv2D(filters=32, strides=[1, 1], kernel_size=[3, 3], padding='same'))
    mo.add(layers.BatchNormalization())
    mo.add(layers.Activation("relu"))
    mo.add(layers.MaxPooling2D(pool_size=(2, 2), strides=[2, 2], padding='same'))

    mo.add(layers.Conv2D(filters=64, strides=[1, 1], kernel_size=[3, 3], padding='same'))
    mo.add(layers.BatchNormalization())
    mo.add(layers.Activation("relu"))
    mo.add(layers.Conv2D(filters=64, strides=[1, 1], kernel_size=[3, 3], padding='same'))
    mo.add(layers.BatchNormalization())
    mo.add(layers.Activation("relu"))
    mo.add(layers.MaxPooling2D(pool_size=(2, 2), strides=[2, 2], padding='same'))

    mo.add(layers.Conv2D(filters=128, strides=[1, 1], kernel_size=[3, 3], padding='same'))
    mo.add(layers.BatchNormalization())
    mo.add(layers.Activation("relu"))
    mo.add(layers.Conv2D(filters=128, strides=[1, 1], kernel_size=[3, 3], padding='same'))
    mo.add(layers.BatchNormalization())
    mo.add(layers.Activation("relu"))
    mo.add(layers.MaxPooling2D(pool_size=(2, 2), strides=[2, 2], padding='same'))

    mo.add(layers.Permute((2,1,3)))
    mo.add(layers.TimeDistributed(layers.Flatten()))
    mo.add(layers.Bidirectional(layers.GRU(256,return_sequences=True)))
    mo.add(layers.Dense(units=256, activation='linear', trainable='true'))
    mo.add(layers.Bidirectional(layers.GRU(256,return_sequences=True)))
    mo.add(layers.Dense(units=11, activation='softmax', trainable='true'))



    # mo.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    mo.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    print(mo.summary())
    return mo
    """
    input = Input(shape=(46, None, 1), name='the_input', dtype='float32')
    m = Conv2D(32, kernel_size=(3, 3), padding='same', name='conv1')(input)
    m = BatchNormalization(axis=1)(m)
    m = Activation("relu")(m)
    m = Conv2D(32, kernel_size=(3, 3), padding='same', name='conv2')(m)
    m = BatchNormalization(axis=1)(m)
    m = Activation("relu")(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(m)

    m = Conv2D(64, kernel_size=(3, 3), padding='same', name='conv3')(m)
    m = BatchNormalization(axis=1)(m)
    m = Activation("relu")(m)
    m = Conv2D(64, kernel_size=(3, 3), padding='same', name='conv4')(m)
    m = BatchNormalization(axis=1)(m)
    m = Activation("relu")(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), name='pool2')(m)

    m = Conv2D(128, kernel_size=(3, 3), padding='same', name='conv5')(m)
    m = BatchNormalization(axis=1)(m)
    m = Activation("relu")(m)
    m = Conv2D(128, kernel_size=(3, 3), padding='same', name='conv6')(m)
    m = BatchNormalization(axis=1)(m)
    m = Activation("relu")(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), name='pool3')(m)

    m = Permute((2, 1, 3), name='permute')(m)
    m = TimeDistributed(Flatten(), name='timedistrib')(m)

    m = Bidirectional(GRU(256, return_sequences=True), name='blstm1')(m)
    m = Dense(256, name='blstm1_out', activation='linear')(m)
    m = Bidirectional(GRU(256, return_sequences=True), name='blstm2')(m)
    y_pred = Dense(11, name='blstm2_out', activation='softmax')(m)

    basemodel = Model(inputs=input, outputs=y_pred)

    labels = Input(name='the_labels', shape=[None, ], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    mo = Model(inputs=[input, labels, input_length, label_length], outputs=[loss_out])
    # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    # model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
    mo.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='Adam')
    mo.summary()

    return mo, basemodel


def vectorize_sequences(sequences, dimension=11):
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


model, base = creatmodel()
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
dir = r"C:\Users\fyx\Desktop\ruanjianbei\images"
lis = os.listdir(dir)
random.shuffle(lis)
limit = 1000
a = []
sequence = ""
for i in range(0, limit):
    name = lis[i]
    wh = random.randint(0, 3)
    #while name[wh] == "_":
    #    wh = random.randint(0, 3)
    sequence = sequence + name[wh]
    img = cv2.imread(dir + '\\' + name)
    #show(img)
    img = pp.trans(img[:, wh * 30:(wh + 1) * 30, :])
    #img = pp.threshold(img)
    #pp.show(img)
    a.append(img)
print(sequence)
tags = vectorize_sequences(sequence, dimension=11)
a = np.asarray(a)
print(a.shape)
x_val = a[:100]
partial_x_train = a[100:]
y_val = tags[:100]
partial_y_train = tags[100:]
history = model.fit(partial_x_train, partial_y_train, epochs=600, batch_size=300, validation_data=(x_val, y_val))
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
