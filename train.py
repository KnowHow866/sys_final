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
    x_train = x_train.astype('float32') / 255.0
    x_test= x_test.astype('float32') / 255.0
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
        student_second = Student(save_path='%s/%s' % (dir_path, setting.student_save_second))
        Evaluate_record = setting.Evaluate_record
        
        measure(teacher.model, 'Teacher')
        measure(student, 'Student')

        for circle in range(5):
            iter_size = int(len(x_train) / setting.slice_number)
            for idx in range(setting.slice_number):
                x_train_slice = x_train[idx*iter_size : (idx + 1)*iter_size]
                y_train_slice = y_train[idx*iter_size : (idx + 1)*iter_size]

                # teacher
                if circle < 2:
                    teacher.save_history(
                        teacher.model.fit(x_train_slice, y_train_slice, epochs=10, batch_size=setting.batch_size, validation_split = 0.1, verbose=1)
                    )
                    format_plot(
                        [teacher.format_history_by_key('acc')],
                        save_name='Teacher_train_accuracy.png',
                        title='Teacher_train_accuracy'
                    )
                print('Iter (%s, %s)'.ljust(120, '-') % (circle, idx))

                teacher_predictions = teacher.model.predict(x_train_slice)
                # student follow to train
                if setting.student_follow and circle < 3:
                    student.save_history(
                        student.model.fit(x_train_slice, teacher_predictions, epochs=10, batch_size=setting.batch_size, validation_split = 0.1, verbose=1)
                    )
                    format_plot(
                        [student.format_history_by_key('acc')],
                        save_name='Student_training_from_teacher.png',
                        title='Student_training_from_teacher'
                    )

                # student second
                if circle > 1:
                    student_second.save_history(
                        student_second.model.fit(x_train_slice, teacher_predictions, epochs=10, batch_size=setting.batch_size, validation_split = 0.1, verbose=1)
                    )
                    format_plot(
                        [student_second.format_history_by_key('acc')],
                        save_name='Second_student_training_from_teacher.png',
                        title='Second_student_training_from_teacher'
                    )

                # evaluate
                _, t_acc = teacher.model.evaluate(x_test, y_test)
                _, s_acc = student.model.evaluate(x_test, y_test)
                _, s2_acc = student_second.model.evaluate(x_test, y_test)

                Evaluate_record['t_acc'].append(t_acc)
                Evaluate_record['s_acc'].append(s_acc)
                if circle > 1: Evaluate_record['s2_acc'].append(s2_acc)
                else: Evaluate_record['s2_acc'].append(None)

                format_plot(
                    [Evaluate_record['t_acc'], Evaluate_record['s_acc'], Evaluate_record['s2_acc']],
                    legends=['Teacher', 'Student', 'Student_second'],
                    save_name='Evaluate.png',
                    title='Teacher, Student to test data accuracy',
                    xlabel='Slice No'
                )

                teacher.save_model()
                student.save_model()             

            # evaluate accuracy, save picture
            loss, acc = teacher.model.evaluate(x_test, y_test)
            print('Training over'.ljust(120, '-'))
            print('Loss %s' % loss)
            print('Acc %s' % acc)

        # training student_second after teacher is done
        # teacher_predictions = teacher.model.predict(x_train)
        # student_second.save_history(
        #     student_second.model.fit(x_train, teacher_predictions, epochs=10, batch_size=setting.batch_size, validation_split = 0.1, verbose=1)
        # )
        # _, s2_acc = student_second.model.evaluate(y_train, y_test)
        # format_plot(
        #     [student_second.format_history_by_key('acc')],
        #     save_name='Second_student_train_accuracy.png',
        #     title='Second_student_train_accuracy, TestData Acc: %s' % s2_acc
        # )
        
            
    print('Training success, mdoel saved')
    
if __name__ == "__main__":
    main()
