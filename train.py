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
from core.data import (pickle_load, cifar_img_reshape, cifar_label_map, cifar_load, save_img)
from model.alex_model import alexnet
import setting

def main():
    # set params & inint setting
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', help='Set a data file to load, default to read all datas in data/train ,setting.py(train_data)')
    parser.add_argument('-name', help='save model as name')
    args = parser.parse_args()
    setting.dir_init()

    # data preprocess
    if args.data:   datas = args.data
    else:
        datas = [os.path.join(setting.train_data, data) for data in os.listdir(setting.train_data)]
    print(datas)

    # training
    with tf.device('gpu:0'):
        model = alexnet()
        model.summary()
        for data in datas:
            data = pickle_load(data)
            x_train, y_train = cifar_load(data)
            for idx in range(50):
                print('Train in batch number: %d' % idx)
                batch_size = 200
                model.fit(x_train, y_train, epochs=10, steps_per_epoch=32, verbose=1)
        model.save(args.name or '%s/model/save/%s_%s_model.h5' % (dir_path, random.randint(0,10000), datetime.now().strftime('%Y-%m-%d')))
    print('Training success, mdoel saved')
    
if __name__ == "__main__":
    main()
