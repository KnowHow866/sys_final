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
    # set params
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', help='wher data to load')
    args = parser.parse_args()

    # data preprocess
    data = unpickle(args.data)

    x_train = reshape_cifar(data[b'data'][:5000])
    y_train = tf.one_hot(data[b'labels'][:5000], 10)

    # training
    with tf.device('gpu:0'):
        model = alexnet()
        model.summary()
        model.fit(x_train, y_train, epochs=500, steps_per_epoch=16, verbose=1)
        model.save('%s/model_saved/%s_model.h5' % (dir_path, datetime.now().strftime('%Y-%m-%d')))
    print('Training success, model saved')
    
if __name__ == "__main__":
    main()
