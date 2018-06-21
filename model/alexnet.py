from .p_model import Parent_model

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dropout

from tensorflow.python.keras import optimizers

import h5py
import time
import os 

# Net stucture
SGD = optimizers.SGD(lr=0.01, momentum=10, decay=0.0, nesterov=False)
input_shape = (32, 32, 3)
model = Sequential([
    Conv2D(16, (3, 3), input_shape=input_shape, padding='same',
        activation='relu'),
    Conv2D(16, (3, 3), activation='relu', padding='same'),
    Conv2D(16, (3, 3), activation='relu', padding='same'),
    Conv2D(16, (3, 3), activation='relu', padding='same'),
    Conv2D(16, (3, 3), activation='relu', padding='same'),
    Dropout(0.5, noise_shape=None, seed=None),
    MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(32, (3, 3), activation='relu', padding='same',),
    Conv2D(32, (3, 3), activation='relu', padding='same',),
    Conv2D(32, (3, 3), activation='relu', padding='same',),
    Conv2D(32, (3, 3), activation='relu', padding='same',),
    Conv2D(32, (3, 3), activation='relu', padding='same',),
    Dropout(0.5, noise_shape=None, seed=None),
    MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(loss='categorical_crossentropy',
        optimizer=SGD,
        metrics=['accuracy'])

class Alexnet(Parent_model):
    def __init__(self, save_path = None):
        super(Alexnet, self).__init__(model, save_path)

