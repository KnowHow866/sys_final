# 3rd module
import tensorflow as tf

# native module
import random
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

# local module
from model.alpha import Alpha
from model.beta import Beta
from model.gamma import Gamma
# local module
import setting

def split_line():
    print('\n')
    print(''.ljust(120, '*'))

alpha = Alpha(save_path='%s/%s' % (dir_path, setting.teacher_save))
alpha.model.summary()

split_line()

beta = Beta(save_path='%s/%s' % (dir_path, setting.student_save))
beta.model.summary()

split_line()

gamma = Gamma(save_path='%s/%s' % (dir_path, 'model/save/gamma'))
gamma.model.summary()


# backend analysis
from tensorflow import keras as keras
K = keras.backend
# print(dir(Student))
# print(dir(K))
# for l in student.layers:
#     print('Layer analysis'.ljust(60, '.'))
#     print(l)
#     print(type(l))
#     print(l.weights)
#     print(l.variables)
#     print(l.output)
#     print()
