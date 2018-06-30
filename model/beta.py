from .p_model import Parent_model

# tf modeul
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dropout

# 3rd modeul
import h5py

# native module
import time
import os 

def get_model():
    input_shape = (32, 32, 3)
    model = Sequential([
        Conv2D(32, (3, 3), input_shape = input_shape, padding = 'same', activation = 'relu'),
        Dropout(0.2, noise_shape=None, seed=None),
        MaxPool2D(pool_size = (2, 2), strides = (2, 2)),

        Flatten(),
        Dropout(0.2, noise_shape=None, seed=None),
        Dense(50, activation = 'relu'),
        Dropout(0.2, noise_shape=None, seed=None),
        Dense(10, activation = 'softmax')
    ])
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model


class Beta(Parent_model):
    def __init__(self, save_path = None):
        super(Beta, self).__init__(get_model, save_path)

