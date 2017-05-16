import os
import glob
import numpy as np

def read_ceps(ver, base_dir = os.getcwd()):
    train_x, train_y = [], []
    test_x, test_y = [], []
    base_dir = base_dir.replace("realtime_analyze", "pyduino")
    train_x, train_y =  make_arr(base_dir, "train" + ver)
    test_x, test_y = make_arr(base_dir, "test" + ver)
    return train_x, train_y, test_x, test_y

def make_arr(base_dir, mode):
    x, y = [], []
    name_list = make_namelist(base_dir, mode)
    for label,name in enumerate(name_list):
        for fn in glob.glob(os.path.join(base_dir, mode, name, "*.ceps.npy")):
            ceps = np.load(fn)
            print(mode, label, ceps, fn)
            num_ceps = len(ceps)
            x.append(ceps)
            y.append(label)
    return np.array(x),np.array(y)

def make_namelist(base_dir, mode):
    name_list = []
    folders = os.listdir(os.path.join(base_dir, mode))
    for f in folders:
        name_list.append(str(f))
    return name_list
