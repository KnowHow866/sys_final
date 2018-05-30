import tensorflow as tf
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
    tmp = []
    for img_data in picture_arr:
        img_R = img_data[0:1024].reshape((32, 32))
        img_G = img_data[1024:2048].reshape((32, 32))
        img_B = img_data[2048:3072].reshape((32, 32))
        img = np.dstack((img_R, img_G, img_B))
        tmp.append(img)
    return tmp

def label_map(lable_arr):
    lable_dict = {
        1: 'airplane',
        2: 'automobile',
        3: 'bird',
        4: 'cat',
        5: 'deer',
        6: 'dog',
        7: 'frog',
        8: 'horse',
        9: 'ship',
        10: 'truck'
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
    # save test
    # for idx in range(30):
    #     img_R = data[b'data'][idx][0:1024].reshape((32, 32))
    #     img_G = data[b'data'][idx][1024:2048].reshape((32, 32))
    #     img_B = data[b'data'][idx][2048:3072].reshape((32, 32))
    #     img = np.dstack((img_R, img_G, img_B))
    #     save_img(img, str(data[b'labels'][idx]))
    # print('Save success')

    model = alexnet()
    model.summary()

    with tf.device('/cpu:0'):
        model.fit(reshape_cifar(data[b'data']), label_map(data[b'labels']), epochs=10, steps_per_epoch=32)
        model.save('%s/model_saved/%s_model.h5' % (dir_path, datetime.now().strftime('%Y-%m-%d')))

    
if __name__ == "__main__":
    main()
