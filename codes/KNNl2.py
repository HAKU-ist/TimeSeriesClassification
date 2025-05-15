from fastdtw import fastdtw
import numpy as np
import os

def load_sequence(path):
    return np.loadtxt(path)

def abs_distance(x,y):
    return np.sum(np.abs(np.array(x)-np.array(y)))

def euclidean_distance(x,y):
    return np.linalg.norm(np.array(x)-np.array(y))

ref1 = load_sequence("../dataset/level2/reference/1.dat")
ref2 = load_sequence("../dataset/level2/reference/2.dat")

dir_path = "../dataset/level2/test"
print("--using euclidean distance--")

for file in os.listdir(dir_path):
    test_seq = load_sequence(os.path.join(dir_path,file))

    #d1,_ = fastdtw(test_seq,ref1,dist=abs_distance)
    #d2,_ = fastdtw(test_seq,ref2,dist=abs_distance)

    d1, _ = fastdtw(test_seq, ref1, dist=euclidean_distance)
    d2, _ = fastdtw(test_seq, ref2, dist=euclidean_distance)

    pred_class = 1 if d1<d2 else 2

    print(f"{file} class {pred_class}")

print("--using abs distance--")
for file in os.listdir(dir_path):
    test_seq = load_sequence(os.path.join(dir_path,file))

    d1,_ = fastdtw(test_seq,ref1,dist=abs_distance)
    d2,_ = fastdtw(test_seq,ref2,dist=abs_distance)

    #d1, _ = fastdtw(test_seq, ref1, dist=euclidean_distance)
    #d2, _ = fastdtw(test_seq, ref2, dist=euclidean_distance)

    pred_class = 1 if d1<d2 else 2

    print(f"{file} class {pred_class}")