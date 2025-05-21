import os
import shutil
import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy import signal
from collections import Counter
from multiprocessing import Pool,cpu_count,Process
from draw_utils import draw_points


if __name__=='__main__':
    det1_path = 'E:/07Experiment-data/VideoSets/03AerialVideos-VehicleVerificationTexts/exp0607/exp1/exp1_corrected.txt'
    det2_path = 'E:/07Experiment-data/VideoSets/03AerialVideos-VehicleVerificationTexts/exp0607/exp2/exp2_corrected.txt'
    path_name = os.path.abspath(det1_path).split('\\')[-3]

    data1 = np.loadtxt(det1_path, dtype=str, delimiter=',')
    data1 = data1.astype(np.float64)
    data2 = np.loadtxt(det2_path, dtype=str, delimiter=',')
    data2 = data2.astype(np.float64)

    data1[:,6]=0  ## car 类别号为0
    data2[:,6]=1  ## truck 类别号为1

    data2[:,1]=data2[:,1]+data1[:,1].max()

    data3 = np.concatenate([data1,data2],axis=0)
    data3 = data3[np.argsort(data3[:, 0]), :]

    # save_path = 'E:/07Experiment-data/VideoSets/03AerialVideos-VehicleVerificationTexts/exp0604/' +path_name+ '.txt'
    outputs_dir = r"./outputs/"
    for f in os.listdir(outputs_dir):
        filepath = os.path.join(outputs_dir, f)
        if os.path.isfile(filepath):
            os.remove(filepath)
        elif os.path.isdir(filepath):
            shutil.rmtree(filepath, True)
    save_path='./outputs/'+path_name+'.txt'
    np.savetxt(save_path, data3, fmt="%d,%d,%.3f,%.3f,%.3f,%.3f,%d,%d,%d,%d", delimiter="\n")