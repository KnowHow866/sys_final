
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
from core.debug import (log, msg)
import setting

def main():
    # set params
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', help='where model to load')
    parser.add_argument('-test', help='use test data to evaluate model')
    parser.add_argument('-test_max_size', help='define max size of test data')
    args = parser.parse_args()
    setting.dir_init()

    with tf.device('cpu:1'):
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
            x_data, y_data = cifar_load(data, start_idx = 0, end_idx = 100)
            print('Start evaluate model')
            (loss, accuracy) = model.evaluate(x_data, y_data, steps = 10)
            print(accuracy) 

            # (accuracy, update_op) = tf.metrics.accuracy(y_data, teacher.model.predict(x_data))
            # init = (tf.global_variables_initializer(), tf.local_variables_initializer())
            # with tf.Session() as sess:
            #             sess.run(init)
            #             print(sess.run(accuracy))
            #             print(sess.run(update_op))

if __name__ == "__main__":
    main()
