import keras
import numpy as np
import cv2
from keras.models import *
from keras.layers import *
from DataProcess import DataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
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


model, base = creatmodel()
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
tdir = r"C:\Users\fyx\Desktop\ruanjianbei\images"
vdir = r"C:\Users\fyx\Desktop\ruanjianbei\images"
traindata = DataGenerator(tdir, [46, 120], 300)
traindata.build_data()
valdata = DataGenerator(vdir, [46, 120], 300)
valdata.build_data()

early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=4, mode='min', verbose=1)
checkpoint = ModelCheckpoint(filepath='crnn--{epoch:02d}--{val_loss:.3f}.hdf5', monitor='loss', verbose=1, mode='min',
                             period=1)
history = model.fit_generator(generator=traindata.next_batch(),
                              steps_per_epoch=int(traindata.n / traindata.batch_size),
                              epochs=30,
                              callbacks=[checkpoint],
                              validation_data=valdata.next_batch(),
                              validation_steps=int(valdata.n / valdata.batch_size))

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
