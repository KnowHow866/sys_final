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
from core.measure import measure
from model.alexnet import Alexnet
from model.student import Student
import setting

def main():
    # set params
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', help='Set a data file to load, default to read all datas in data/train ,setting.py(train_data)')
    parser.add_argument('-name', help='save model as name')
    args = parser.parse_args()
    # init setting
    setting.dir_init()
    batch_size = setting.batch_size or 100
    student_follow = setting.student_follow or 20
    token = setting.token()

    # dataset paths
    if args.data:   datas = [args.data]
    else:
        datas = [os.path.join(setting.train_data, data) for data in os.listdir(setting.train_data)]
    print(datas)

    # training
    with tf.device('gpu:0'):
        teacher = Alexnet()
        student = Student()
        save_as = args.name or '%s_%s.h5' % (random.randint(0,10000), datetime.now().strftime('%Y-%m-%d'))
        
        measure(teacher.model, 'Teacher')
        measure(student, 'Student')

        # iter each dataset
        for data_idx, data in enumerate(datas):
            path_label = data
            data = pickle_load(data)
            # train in batch
            for batch_number in range(100):
                print('Train in batch number: %d'.ljust(30, '-') % batch_number)
                x_batch, y_batch = cifar_load(data, start_idx = (batch_number * batch_size), end_idx = (batch_number + 1) * batch_size)

                if (token.teacher_turn):
                    teacher.model.fit(x_batch, y_batch, epochs=10, steps_per_epoch=32, verbose=1)
                    
                if (token.student_turn):
                    y_batch = teacher.model.predict(x_batch)
                    student.fit(x_batch, y_batch, epochs=10, steps_per_epoch=32, verbose=1)

            # save point
            teacher.model.save('%s/model/save/%s_%s' % (dir_path, data_idx, save_as))
        teacher.model.save('%s/model/save/%s' % (dir_path, save_as))
    print('Training success, mdoel saved')
    
if __name__ == "__main__":
    main()
