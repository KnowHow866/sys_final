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
        save_as = args.name or '%s_%s.h5' % (random.randint(0,10000), datetime.now().strftime('%Y-%m-%d'))
        for data_idx, data in enumerate(datas):
            data = pickle_load(data)
            # x_train, y_train = cifar_load(data, 100)
            for batch_number in range(50):
                print('Train in batch number: %d' % batch_number)
                batch_size = 200
                x_batch, y_batch = cifar_load(data, start_idx = (batch_number * batch_size), end_idx = (batch_number + 1) * batch_size)
                # x_batch = x_train[(batch_number * batch_size) : (batch_number + 1) * batch_size]
                # y_batch = y_train[(batch_number * batch_size) : (batch_number + 1) * batch_size]
                model.fit(x_batch, y_batch, epochs=10, steps_per_epoch=32, verbose=1)
            model.save('%s/model/save/%s_%s' % (dir_path, data_idx, save_as))
        model.save('%s/model/save/%s' % (dir_path, save_as))
    print('Training success, mdoel saved')
    
if __name__ == "__main__":
    main()
