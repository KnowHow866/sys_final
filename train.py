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
from core.measure import (measure, calculate_accuracy, calculate_prediction_match_rate, find_prediction, draw_line_graph)
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

        print(len(x_train))

        history = teacher.model.fit(x_train, y_train, epochs=10, batch_size=16, validation_split = 0.1, verbose=1)
        # evaluate accuracy, save picture
        loss, acc = teacher.model.evaluate(x_test, y_test)
        print('Training over'.ljust(120, '-'))
        print('Loss %s' % loss)
        print('Acc %s' % acc)
        print(history.history.keys())

        plot(history.history['acc'])

        # if True:
        #     trained_batches += setting.snapshop_default

        #     # training evaluate
        #     teacher.save_record((trained_batches, calculate_accuracy(teacher.model.predict(test_x), test_labels )))
        #     student.save_record((trained_batches, calculate_accuracy(student.model.predict(test_x), test_labels )))
        #     draw_line_graph([
        #         teacher.format_record('Teacher'),
        #         student.format_record('Student'),
        #     ], save_name = 'accuracy.png')

        #     # student learning evaluate
        #     student.save_match_teacher((trained_batches, calculate_prediction_match_rate(
        #         student.model.predict(test_x),
        #         teacher.model.predict(test_x)
        #     )
        #     ))
        #     draw_line_graph([student.format_match_teacher('Student')],
        #         title = 'Student prediction match teacher\'s accuracy',
        #         save_name = 'student_match_teacher.png'
        #     )

        # if (token.teacher_turn()):
        #     teacher.model.fit(x_batch, y_batch, epochs=10, steps_per_epoch=32, validation_split = 0.1, verbose=1)
            
        # if (token.student_turn()):
        #     teacher.model.fit(x_batch, y_batch, epochs=10, steps_per_epoch=64, validation_split = 0.1, verbose=1)
        #     predict_batch = teacher.model.predict(x_batch)
        #     student.model.fit(x_batch, predict_batch, epochs=16, steps_per_epoch=128, validation_split = 0.1, verbose=1)
            
    teacher.save_model()
    student.save_model()
    print('Training success, mdoel saved')
    
if __name__ == "__main__":
    main()
