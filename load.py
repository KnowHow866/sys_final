
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

def main():
    # set params
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', help='where model to load')
    parser.add_argument('-testdata', help='use test data to evaluate model')
    parser.add_argument('-testdata_max_size', help='define max size of test data')
    args = parser.parse_args()

    # summary model model
    if args.model is None:
        print('No model path th run')
        sys.exit(0)
    model = keras.models.load_model(args.model)
    model.summary()

    # if there is testint dataset, evslutate model
    if args.testdata:
        data = pickle_load(args.testdata)
        x_data, y_data = cifar_load(data, int(args.testdata_max_size))
        print('Start evaluate model')
        evaluate = model.evaluate(x_data, y_data, steps = 10)
        print(evaluate) 

if __name__ == "__main__":
    main()
