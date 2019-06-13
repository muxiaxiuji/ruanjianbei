import cv2
import numpy as np
import tensorflow as tf
defaultsize = [46, 30]


def minus(a, b):
    if a > b:
        return a - b
    else:
        return b - a


def isvalid(x, y, size):
    return 0 <= x < size[0] and 0 <= y < size[1]


def show(img):
    cv2.namedWindow("Image")
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def trans(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = img.shape
    """
    ret, th1 = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
    ans = th2
    ans=cv2.erode(ans,np.ones((3,3),))
    """
    newimg = np.zeros([size[0], size[1]],np.uint8)
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            sq=[]
            step = 1
            for x in range(i - step, i + step + 1):
                for y in range(j - step, j + step + 1):
                    if isvalid(x, y, size):
                        sq.append(minus(img[i, j], img[x, y]))
            sq.sort(reverse=True)
            newimg[i, j] = (int(sq[0])+int(sq[1]))/2
    return newimg.reshape([size[0], size[1], 1]).astype(np.float32)


def threshold(img):
    avg = 0
    ma = 0
    size = img.shape
    for x in img:
        for y in x:
            for z in y:
                avg += z
                ma=max(ma,z)
    avg = avg/size[0]/size[1]/size[2]
    #ret, img = cv2.threshold(img, int(avg), 255, cv2.THRESH_BINARY)
    #print(ma)
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            """
            if img[i, j]<avg:
                img[i, j] = 0
            else:
                img[i,j] = 255
            img[i, j] = 255-img[i, j]
            """
            img[i, j] = ma-img[i, j]
            #print(img[i, j],end=" ")
        #print()
    return img

a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
c = a + b
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))