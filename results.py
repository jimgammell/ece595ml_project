import numpy as np

from utils import log_print as print

class Results:
    def __init__(self):
        self.data = {}
    def keys(self):
        return self.data.keys()
    def add_keys(self, *keys):
        for key in keys:
            self.data.update({key: {'epochs': [], 'values': []}})
    def append_value(self, epoch, key, value):
        self.data[key]['epochs'].append(epoch)
        self.data[key]['values'].append(value)
    def update(self, epoch, train_dict=None, val_dict=None, test_dict=None, print_res=True):
        print('Epoch {} complete.'.format(epoch))
        for prefix, d in {'train_': train_dict, 'val_': val_dict, 'test_': test_dict}.items():
            if d == None:
                continue
            if print_res:
                print('{}:'.format(prefix[:-1]))
            for key, item in d.items():
                key = prefix + key
                if not key in self.keys():
                    self.add_keys(key)
                self.append_value(epoch, key, item)
                if print_res:
                    print('\t{}: {}'.format(key, np.sum(item) if type(item) == list else item))
    def add_single_pair(self, key, value):
        assert not key in self.keys()
        self.data.update({key: value})
                
    def get_traces(self, key):
        x = np.array(self.data[key]['epochs'])
        y = np.array(self.data[key]['values'])
        return x, y


# Mean function which returns nan when it encounters empty list, and ignores nan in list.
def mean(x):
    if len(x) == 0:
        return np.nan
    return np.nanmean(x)
