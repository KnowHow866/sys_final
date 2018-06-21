# 3rd module
import tensorflow as tf
from pympler import asizeof
import matplotlib.pyplot as plt

def measure(data = None, name = 'NAN'):
    '''Format print an bytes size of input object'''
    with tf.device('cpu:1'):
        if data is None:
            raise Exception('None data can not be measure')
        print('Measure size: %s'.ljust(60, '.') %  name)
        size = asizeof.asizeof(data)
        print('\t%s bytes, \t%s KB, \t%s MB' % (size, size / 1024, size / 1048576))
        print()

def draw_line_graph(datas = None, save_path = None):
    if datas is None: raise Exception('Datas must given')
    if len(datas) > 3: raise Exception('3 data is the max')
    if save_path is None: raise Exception('Please give save path')

    plt.title('Training accuarcy')
    plt.xlabel('Batch numbers')
    plt.ylabel('Accuracy')

    labels = [data['lable'] for data in datas]
    x = [data['x'] for data in datas]
    y = [data['y'] for data in datas]
    colors = ['r', 'b', 'g']
    styles = ['o', 's', '^']

    for idx, label in enumerate(labels):
        plt.plot(x[idx], y[idx], label = label, color = colors[idx], marker = styles[idx])

    plt.legend(loc = 'best')
    plt.savefig('%s/accuracy.png'  % save_path)
    plt.close()
