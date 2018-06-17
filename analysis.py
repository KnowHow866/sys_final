# 3rd module
import tensorflow as tf

# local module
from model.alexnet import Alexnet
from model.student import Student

print('Teacher model'.ljust(60, '.'))
teacher = Alexnet()
teacher.summary()

print(''.ljust(120, '*'))

print('Student model'.ljust(60, '.'))
student = Student()
student.summary()

# backend analysis
from tensorflow import keras as keras
K = keras.backend
# print(dir(Student))
# print(dir(K))
for l in student.layers:
    print('Layer analysis'.ljust(60, '.'))
    print(l)
    print(type(l))
    print(l.weights)
    print(l.variables)
    print(l.output)
    print()
