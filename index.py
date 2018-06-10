# 3rd module
import tensorflow as tf
import numpy as np
import argparse
import h5py
import numpy as np

# native module
import random
from datetime import datetime
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

# set matplotlib for linux
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# local module
from core.data import (pickle_load, cifar_img_reshape, cifar_label_map, save_img)
from model.alex_model import alexnet

def main():
    # set params
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', help='where data to load')
    args = parser.parse_args()

    # data preprocess
    data = pickle_load(args.data)

    # training
    with tf.device('gpu:0'):
        model = alexnet()
        model.summary()
        for idx in range(50):
            print('Train in batch number: %d' % idx)
            batch_size = 200
            x_train = cifar_img_reshape(data[b'data'][idx * batch_size : (idx + 1) * batch_size])
            y_train = tf.one_hot(data[b'labels'][idx * batch_size : (idx + 1) * batch_size], 10)
            model.fit(x_train, y_train, epochs=10, steps_per_epoch=32, verbose=1)
        model.save('%s/model/save/%s_%s_model.h5' % (dir_path, random.randint(0,10000), datetime.now().strftime('%Y-%m-%d')))
    print('Training success, mdoel saved')
    
if __name__ == "__main__":
    main()
