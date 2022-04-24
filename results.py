class Results:
    def __init__(self, *keys):
        self.data = {}
        for key in keys:
            self.data.update({key: {'epoch': [], 'value': []}})
    def append_pref(self, epoch, data, pref):
        for key in data.keys():
            key = pref + key
            assert key in self.data.keys
            self.data[key]['epoch'].append(epoch)
            self.data[key]['value'].append(data[key])
    def append(self, epoch, train_data, test_data):
        self.append_pref(epoch, train_data, 'train_')
        self.append_pref(epoch, test_data, 'test_')
    def get_trace(self, key):
        X = self.data[key]['epoch']
        Y = self.data[key]['value']
        return X, Y
    def keys(self):
        return self.data.keys()