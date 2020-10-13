import os
import logging
import numpy as np

def make_dir(dirs):
    try:
        if not os.path.exists(dirs):
            os.makedirs(dirs)
    except Exception as err:
        print("create_dirs error!")
        exit()

class AverageMeter:
    """
    Class to be an average meter for any average metric like loss, accuracy,
    etc...
    """

    def __init__(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def val(self):
        return self.avg






