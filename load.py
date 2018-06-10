
# 3rd module
import tensorflow as tf
import numpy as np
import argparse
import pickle
import h5py
import numpy as np
from tensorflow import keras as keras
import pickle

# native module
import sys as sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

# local module
from core.data import (pickle_load, cifar_img_reshape, cifar_label_map, cifar_load, save_img)
import setting

def main():
    # set params
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', help='where model to load')
    parser.add_argument('-test', help='use test data to evaluate model')
    parser.add_argument('-test_max_size', help='define max size of test data')
    args = parser.parse_args()
    setting.dir_init()

    # summary model model
    try:
        model = keras.models.load_model(args.model or setting.load_model)
        model.summary()
    except Exception as err:
        print(err)
        print('No model found to load')
        sys.exit(0)

    # if there is testint dataset, evslutate model
    if args.test:
        data = pickle_load(args.test)
        x_data, y_data = cifar_load(data, max_size =int(args.test_max_size))
        print('Start evaluate model')
        evaluate = model.evaluate(x_data, y_data, steps = 10)
        print(evaluate) 

if __name__ == "__main__":
    main()
