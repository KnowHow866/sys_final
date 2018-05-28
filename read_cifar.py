import argparse
import pickle
import os
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

def unpickle(path):
    with open(path, 'rb') as file:
        tmp = pickle.load(file, encoding='bytes')
    return tmp

def save_img(img, label='Default'):
    plt.clf()
    plt.title(label)
    plt.imshow(img)
    plt.savefig('images/%s_%s.jpg' % (label, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', help='wher data to load')
    args = parser.parse_args()

    data = unpickle(args.data)
    # save test
    save_img(np.reshape(data[b'data'][0], (32, 32, 3)), data[b'labels'][0])
    print('Save success')

    
if __name__ == "__main__":
    main()
