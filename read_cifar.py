import argparse
import pickle
import os
from datetime import datetime
import numpy as np
import random

# set matplotlib for linux
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))

def unpickle(path):
    with open(path, 'rb') as file:
        tmp = pickle.load(file, encoding='bytes')
    return tmp

def save_img(img, label='Default'):
    print('Ready to save image: %s' % label)
    print(img)

    plt.title(label)
    plt.imshow(img)
    plt.savefig('images/%s_%s_%s.jpg' % (label, datetime.now().strftime('%Y-%m-%d'), random.randint(0,10000)))
    plt.clf()
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', help='wher data to load')
    args = parser.parse_args()

    data = unpickle(args.data)
    # save test
    for idx in range(30):
        save_img(np.reshape(data[b'data'][idx], (32, 32, 3)), str(data[b'labels'][idx]))
    print('Save success')

    
if __name__ == "__main__":
    main()
