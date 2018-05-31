from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras.layers import Flatten

import h5py
import time
import os 

# Alex structure
def alexnet():
    input_shape = (32, 32, 3)
    model = Sequential([
        Conv2D(16, (3, 3), input_shape=input_shape, padding='same',
            activation='relu'),
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        Conv2D(16, (3, 3), activation='relu', padding='same',),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same',),
        Conv2D(32, (3, 3), activation='relu', padding='same',),
        Conv2D(32, (3, 3), activation='relu', padding='same',),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same',),
        Conv2D(32, (3, 3), activation='relu', padding='same',),
        Conv2D(32, (3, 3), activation='relu', padding='same',),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same',),
        Conv2D(64, (3, 3), activation='relu', padding='same',),
        Conv2D(64, (3, 3), activation='relu', padding='same',),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    return model
