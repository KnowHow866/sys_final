import tensorflow as tf
import numpy as np
import argparse
import pickle
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
from datetime import datetime
import h5py
import time
import numpy as np
import random

# set matplotlib for linux
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# model
from model.alex_model import alexnet

dir_path = os.path.dirname(os.path.realpath(__file__))

def unpickle(path):
    with open(path, 'rb') as file:
        tmp = pickle.load(file, encoding='bytes')
    return tmp

def reshape_cifar(picture_arr):
    tmp = None
    for idx in range(len(picture_arr)):
        img_R = picture_arr[idx][0:1024].reshape((32, 32))
        img_G = picture_arr[idx][1024:2048].reshape((32, 32))
        img_B = picture_arr[idx][2048:3072].reshape((32, 32))
        img = np.dstack((img_R, img_G, img_B))
        if tmp is None:
            tmp = np.array([img], np.float32)
        else:
            tmp = np.append(tmp, np.array([img], np.float32), axis=0)

        if (idx % 50) == 0:
            print('Picture decode: %f percent' % (idx / len(picture_arr)))

    return tmp

def label_map(lable_arr):
    lable_dict = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }
    for idx in range(len(lable_arr)):
        lable_arr[idx] = lable_dict.get(lable_arr[idx], lable_arr[idx])
    return lable_arr
    

def save_img(img, label='Default'):
    print('Ready to save image: %s' % label)
    print(img)

    plt.title(label)
    plt.imshow(img)
    plt.savefig('images/%s_%s_%s.jpg' % (label, datetime.now().strftime('%Y-%m-%d'), random.randint(0,10000)))
    plt.clf()
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', help='wher data to load')
    args = parser.parse_args()

    data = unpickle(args.data)

    # selef made model
    import tensorflow as tf
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import Dense
    from tensorflow.python.keras.layers import Conv2D
    from tensorflow.python.keras.layers import MaxPool2D
    from tensorflow.python.keras.layers import Flatten

    # model = Sequential()

    # model.add(Conv2D(16, 3, 3, input_shape=(32, 32, 3)))
    # model.add(MaxPool2D(2, 2))
    # model.add(Flatten())
    # model.add(Dense(units=32, activation='relu'))
    # model.add(Dense(units=10, activation='softmax'))

    # model.compile(loss='categorical_crossentropy',
    #             optimizer='adam',
    #             metrics=['accuracy'])
    model = alexnet()
    model.summary()


    x_train = reshape_cifar(data[b'data'][:500])
    y_train = tf.one_hot(data[b'labels'][:500], 10)

    with tf.device('gpu:0')
        model.fit(x_train, y_train, epochs=10, steps_per_epoch=32, verbose=1)
        model.save('%s/model_saved/%s_model.h5' % (dir_path, datetime.now().strftime('%Y-%m-%d')))
    
if __name__ == "__main__":
    main()
