# 3rd module
import tensorflow as tf
import numpy as np
import argparse
import h5py
import numpy as np

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
from core.measure import (measure, draw_line_graph)
from model.alexnet import Alexnet
from model.student import Student
import setting

def main():
    # set params
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', help='Set a data file to load, default to read all datas in data/train ,setting.py(train_data)')
    args = parser.parse_args()
    # init setting
    setting.dir_init()
    batch_size = setting.batch_size or 100
    student_follow = setting.student_follow or 20
    token = setting.token()
    snapshop_token = setting.snapshop_token()

    # dataset paths
    if args.data:   datas = [args.data]
    else:
        datas = [os.path.join(setting.train_data, data) for data in os.listdir(setting.train_data)]
    print(datas)

    # training
    with tf.device('gpu:0'):
        save_as = '%s_%s.h5' % (random.randint(0,10000), datetime.now().strftime('%Y-%m-%d'))
        teacher = Alexnet(save_path='%s/%s' % (dir_path, setting.teacher_save))
        student = Student(save_path='%s/%s' % (dir_path, setting.student_save))
        
        measure(teacher.model, 'Teacher')
        measure(student, 'Student')

        # iter each dataset
        trained_batches = 0
        for data_idx, data in enumerate(datas):
            path_label = data
            data = pickle_load(data)

            for batch_number in range(100):
                print('Train in batch number: %d'.ljust(30, '-') % batch_number)
                x_batch, y_batch = cifar_load(data, start_idx = (batch_number * batch_size), end_idx = (batch_number + 1) * batch_size)

                # evaluate accuracy, save picture
                if snapshop_token.check() is True:
                    trained_batches += setting.snapshop_default
                    teacher.save_record((trained_batches, teacher.model.evaluate(x_batch, y_batch, verbose = 1)))
                    # student.save_record((trained_batches, student.model.evaluate(x_batch, y_batch, steps = 10)))
                    draw_line_graph([
                        teacher.format_record('Teacher'),
                        student.format_record('Student'),
                    ], 'images')

                # if (token.teacher_turn()):
                teacher.model.fit(x_batch, y_batch, epochs=10, steps_per_epoch=32, verbose=1)
                    
                # if (token.student_turn()):
                #     teacher.model.fit(x_batch, y_batch, epochs=10, steps_per_epoch=64, verbose=1)
                #     predict_batch = teacher.model.predict(x_batch)
                #     student.model.fit(x_batch, predict_batch, epochs=32, steps_per_epoch=128, verbose=1)
                    
            teacher.save_tmp()
            # student.save_tmp()
        teacher.save_model()
        # student.save_model()
    print('Training success, mdoel saved')
    
if __name__ == "__main__":
    main()
