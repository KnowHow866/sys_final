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
        img_R = data[b'data'][idx][0:1024].reshape((32, 32))
        img_G = data[b'data'][idx][1024:2048].reshape((32, 32))
        img_B = data[b'data'][idx][2048:3072].reshape((32, 32))
        img = np.dstack((img_R, img_G, img_B))
        save_img(img, str(data[b'labels'][idx]))
    print('Save success')

    
if __name__ == "__main__":
    main()
