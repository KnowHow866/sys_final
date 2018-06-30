
from tensorflow.python.keras.models import Sequential

import h5py
import time
import os 

class Parent_model:
    def __init__(self, model = None, save_path = None):
        if model is None: raise Exception('Model must given')
        if save_path is None: raise Exception('Save_path must given')

        self.model = model()
        self.accuracy = [] # (batch_number_trained, evaulation)
        self.history = None
        
        self.save_path = save_path
        self.tmp_path = '/'.join([save_path, 'tmp'])
        self.save_times = 0 # tmp save times

        # init dir
        if not os.path.exists(self.save_path): os.mkdir(self.save_path)
        if not os.path.exists(self.tmp_path): os.mkdir(self.tmp_path)
        # test save
        self.model.save('%s/init.h5' % self.tmp_path)

        print('Model init'.ljust(30, '*'))
        print(self.model)

    def save_tmp(self):
        self.model.save('%s/tmp_%s.h5' % (self.tmp_path, self.save_times))
        self.save_times += 1

    def save_model(self):
        self.model.save('%s/model.h5' % self.save_path)

    def save_record(self, record):
        self.accuracy.append(record)

    def format_record(self, label = None):
        if label is None: raise Exception('Please give label')
        return {
            'lable': label,
            'x': [item[0] for item in self.accuracy],
            'y': [item[1] for item in self.accuracy]
        }

    def save_history(self, history = None):
        if history is None: raise Exception('Params not enough')

        if self.history is None:
            self.history = history
        else:
            keys = ['val_loss', 'val_acc', 'loss', 'acc']
            for key in keys:
                self.history.history[key] += history.history[key]


    def format_history_by_key(self, key):
        if key is None: raise Exception('Params not enough')
        if self.history is None: raise Exception('NO history saved')

        # print('format_history called'.ljust(60, '.'))
        # print('Now %s len: %s' % (key, len(self.history.history[key])))
        print(self.history.history[key])
        return self.history.history[key]

    def save_match_teacher(self, record):
        self.match_teacher.append(record)

    def format_match_teacher(self, label = None):
        if label is None: raise Exception('Please give label')
        return {
            'lable': label,
            'x': [item[0] for item in self.match_teacher],
            'y': [item[1] for item in self.match_teacher]
        }