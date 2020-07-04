# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import yaml
from easydict import EasyDict as edict

def Config(filename):
    listfile1 = open(filename, 'r')
    listfile2 = open(filename, 'r')
    cfg = edict(yaml.safe_load(listfile1))
    settings_show = listfile2.read().splitlines()

    listfile1.close()
    listfile2.close()

    return cfg, settings_show