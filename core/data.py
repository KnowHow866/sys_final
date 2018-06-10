# 3rd pk
import tensorflow as tf
import pickle
import numpy as np

# native pk
import random
from datetime import datetime

def pickle_load(path):
    '''open an file with pickle '''
    with open(path, 'rb') as file:
        tmp = pickle.load(file, encoding='bytes')
    return tmp

def cifar_img_reshape(picture_arr):
    """reshape cifar-10 datasets origin data to format (x, 32, 32, 3)"""
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
            print('Picture decode: %d percent' % (idx / len(picture_arr)*100))

    return tmp

def cifar_load(data, max_size = None):
    """Load data from cifar"""
    print(max_size)
    print(type(max_size))
    x_data = cifar_img_reshape(data[b'data'][:max_size])
    y_data = tf.one_hot(data[b'labels'][:max_size], 10)
    return (x_data, y_data)

def cifar_label_map(lable_arr):
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
    """Save image to dir images/"""
    print('Ready to save image: %s' % label)
    print(img)

    plt.title(label)
    plt.imshow(img)
    plt.savefig('images/%s_%s_%s.jpg' % (label, datetime.now().strftime('%Y-%m-%d'), random.randint(0,10000)))
    plt.clf()
    pass
    