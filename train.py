# 3rd module
import tensorflow as tf
import numpy as np
import argparse
import h5py
import numpy as np
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.utils import to_categorical

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
from core.debug import (log, msg)
from core.measure import (measure, calculate_accuracy, calculate_prediction_match_rate, find_prediction, draw_line_graph, format_plot, concat_history)
from model.alexnet import Alexnet
from model.student import Student
import setting

def main():
    # set params
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # init setting
    setting.dir_init()
    batch_size = setting.batch_size or 100
    student_follow = setting.student_follow or 20
    token = setting.token()
    snapshop_token = setting.snapshop_token()

    # dataset paths
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    def plot(_data):
        import matplotlib.pyplot as plt
        plt.plot(_data)
        plt.show()

    # training
    with tf.device('gpu:0'):
        save_as = '%s_%s.h5' % (random.randint(0,10000), datetime.now().strftime('%Y-%m-%d'))
        teacher = Alexnet(save_path='%s/%s' % (dir_path, setting.teacher_save))
        student = Student(save_path='%s/%s' % (dir_path, setting.student_save))
        
        measure(teacher.model, 'Teacher')
        measure(student, 'Student')

        history = None
        for circle in range(10):
            iter_size = int(len(x_train) / setting.slice_number)
            for idx in range(setting.slice_number):
                if not history:
                    history = teacher.model.fit(x_train[idx*iter_size : (idx + 1)*iter_size], y_train[idx*iter_size : (idx + 1)*iter_size], epochs=10, batch_size=setting.batch_size, validation_split = 0.1, verbose=1)
                else:
                    history = concat_history(history, teacher.model.fit(x_train[idx*iter_size : (idx + 1)*iter_size], y_train[idx*iter_size : (idx + 1)*iter_size], epochs=10, batch_size=setting.batch_size, validation_split = 0.1, verbose=1))
                format_plot(history.history['acc'], 'accuracy.png')
                print('Iter (%s, %s)' % (circle, idx))                

        # evaluate accuracy, save picture
        loss, acc = teacher.model.evaluate(x_test, y_test)
        print('Training over'.ljust(120, '-'))
        print('Loss %s' % loss)
        print('Acc %s' % acc)
        print(history.history.keys())

        print(history.history['acc'])
        format_plot(history.history['acc'], 'accuracy.png')
            
    teacher.save_model()
    student.save_model()
    print('Training success, mdoel saved')
    
if __name__ == "__main__":
    main()
