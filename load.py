
# 3rd module
import tensorflow as tf
import numpy as np
import argparse
import pickle
import h5py
import numpy as np
from tensorflow import keras as keras

# native module
import sys as sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

def main():
    # set params
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', help='where model to load')
    args = parser.parse_args()

    # get model
    if args.model is None:
        print('No model path th run')
        sys.exit(0)
    model = keras.models.load_model(args.model)

    model.summary()
        

if __name__ == "__main__":
    main()
