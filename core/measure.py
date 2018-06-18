# 3rd module
import tensorflow as tf
from pympler import asizeof

def measure(data = None, name = 'NAN'):
    '''Format print an bytes size of input object'''
    with tf.device('cpu:1'):
        if data is None:
            raise Exception('None data can not be measure')
        print('Measure size: %s'.ljust(60, '.') %  name)
        size = asizeof.asizeof(data)
        print('\t%s bytes, \t%s KB, \t%s MB' % (size, size / 1024, size / 1048576))
        print()
