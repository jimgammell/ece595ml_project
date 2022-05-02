import random
import numpy as np
import torch

log_file = None

# Print statements will be written to this file, in addition to the terminal.
def create_log_file(dest):
    global log_file
    assert log_file == None
    log_file = open(dest, 'w')

# Print statement can be overwritten with this to simultaneously print to terminal
#  and save to log file.
def log_print(*args, **kwargs):
    global log_file
    assert log_file != None
    print(*args, **kwargs)
    print(*args, file=log_file, **kwargs)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)