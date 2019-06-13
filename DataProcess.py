import cv2
import os, random
import numpy as np
import PictureProcess as pp


def text_to_labels(text):
    ret = []
    for i in range(0,4):
        if text[i] == '_':
            ret.append(10)
        else:
            ret.append(int(text[i]))
    return ret


class DataGenerator:
    def __init__(self, img_dir, img_size,
                 batch_size, max_text_len=20):
        self.img_size = img_size
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.img_dir = img_dir                  # image dir path
        self.img_list = os.listdir(self.img_dir)     # images list
        self.n = len(self.img_list)                      # number of images
        self.indexes = list(range(self.n))
        self.cur = 0
        self.imgs = np.zeros((self.n, self.img_size[0], self.img_size[1], 1))
        self.texts = []

    def build_data(self):
        print(self.n, " Image Loading start...")
        for i, img_file in enumerate(self.img_list):
            img = cv2.imread(self.img_dir + img_file)
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
            img = pp.trans(img)
            self.imgs[i, :, :, :] = img
            self.texts.append(img_file[0:4])
        print(len(self.texts) == self.n)
        print(self.n, " Image Loading finish...")

    def next_sample(self):
        self.cur += 1
        if self.cur >= self.n:
            self.cur = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur]], self.texts[self.indexes[self.cur]]

    def next_batch(self):
        while True:
            X_data = np.ones([self.batch_size, self.img_size[0], self.img_size[1], 1])
            Y_data = np.ones([self.batch_size, self.max_text_len])
            input_length = np.ones((self.batch_size, 1)) * self.img_size[1]  # (bs, 1)
            label_length = np.zeros((self.batch_size, 1))           # (bs, 1)

            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                img = np.expand_dims(img, -1)
                X_data[i] = img
                Y_data[i] = text_to_labels(text)
                label_length[i] = len(text)

            inputs = {
                'the_input': X_data,  # (bs, 128, 64, 1)
                'the_labels': Y_data,  # (bs, 8)
                'input_length': input_length,  # (bs, 1) -> 모든 원소 value = 30
                'label_length': label_length  # (bs, 1) -> 모든 원소 value = 8
            }
            outputs = {'ctc': np.zeros([self.batch_size])}   # (bs, 1) -> 모든 원소 0
            yield (inputs, outputs)
