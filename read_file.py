import h5py
import pandas as pd


data = dict()
with h5py.File('./dataset/prior.h5', 'r') as f:
    # 遍历文件中的所有属性
    for key in f.keys():
        value = f[key][:]
        print(key)
        data[key] = value


# def write_into_vtk(data, file):



