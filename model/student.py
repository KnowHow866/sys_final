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

input_shape = (32, 32, 3)
model = Sequential([
    Conv2D(10, (3, 3), input_shape = input_shape, padding = 'same', activation = 'relu'),
    MaxPool2D(pool_size = (2, 2), strides = (2, 2)),
    Conv2D(10, (3, 3), padding = 'same', activation = 'relu'),
    Dropout(0.5, noise_shape=None, seed=None),
    MaxPool2D(pool_size = (2, 2), strides = (2, 2)),
    Flatten(),
    Dense(16, activation = 'relu'),
    Dense(10, activation = 'softmax')
])
model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])


class Student(Parent_model):
    def __init__(self, save_path = None):
        super(Student, self).__init__(model, save_path)
        self.match_teacher = [] #(acc / batches)

    def save_match_teacher(self, record):
        self.match_teacher.append(record)

    def format_match_teacher(self, label = None):
        if label is None: raise Exception('Please give label')
        return {
            'lable': label,
            'x': [item[0] for item in self.match_teacher],
            'y': [item[1] for item in self.match_teacher]
        }
