# -*- coding: utf-8 -*-

import h5py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models  import Model


def print_structure(weight_file_path, detal = False):
    w = []
    """
    Prints out the structure of HDF5 file.
    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print(" {}: {}".format(key, value))
        
        if len(f.items())==0:
            return 

        for layer, g in f.items():
            print("{}".format(layer))
            print("Attributes: ")
            for key, value in g.attrs.items():
                print(" {}: {}".format(key, value))
                
            for p_name in g.keys():
                param = g[p_name]
                for k_name in param.keys():
                    if detal: 
                        print("Dataset:")
                        print(" {}/{}".format(p_name, k_name, param.get(k_name)[:]))
                        print(" {}".format(param.get(k_name)[:]))
                        w.append(param.get(k_name)[:])
    finally:
        f.close()
    return w

if __name__ == '__main__':
    aaa = print_structure('demo.h5')