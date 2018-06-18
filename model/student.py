# tf modeul
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras.layers import Flatten

# 3rd modeul
import h5py

# native module
import time
import os 

def Student():
    input_shape = (32, 32, 3)
    model = Sequential([
        Conv2D(10, (3, 3), input_shape = input_shape, padding = 'same', activation = 'relu'),
        MaxPool2D(pool_size = (2, 2), strides = (2, 2)),
        Conv2D(10, (3, 3), padding = 'same', activation = 'relu'),
        MaxPool2D(pool_size = (2, 2), strides = (2, 2)),
        Flatten(),
        Dense(10, activation = 'relu'),
        Dense(10, activation = 'softmax')
    ])
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    from pympler import asizeof
    print('Student \t%s' % asizeof.asizeof(model))
    return model
