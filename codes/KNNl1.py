from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
import os

def load_sequence(file_path):
    return np.loadtxt(file_path)

def abs_distance(x,y):
    return abs(x-y)

ref1 = load_sequence("../dataset/level1/reference/1.dat")
ref2 = load_sequence("../dataset/level1/reference/2.dat")

for file in os.listdir("../dataset/level1/test"):
    test_seq = load_sequence(f"../dataset/level1/test/{file}")

    d1,_ = fastdtw(test_seq,ref1,dist = abs_distance)
    d2,_ = fastdtw(test_seq,ref2,dist = abs_distance)

    pred_class = 1 if d1<d2 else 2
    print(f"{file} class {pred_class}")