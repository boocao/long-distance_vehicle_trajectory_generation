
import os

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

# Rauch–Tung–Striebel_smoother
def trajectory_smoothing_by_RTS(input_x,input_y):
    fk = KalmanFilter(dim_x=4, dim_z=2)
    # state transition matrix
    fk.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    # Measurement function
    fk.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0], ])
    # covariance matrix
    fk.P = np.array([[10, 0, 0, 0],
                     [0, 10, 0, 0],
                     [0, 0, 1000, 0],
                     [0, 0, 0, 1000]])
    # state uncertainty
    fk.R = np.array([[10, 0],
                     [0, 10]])
    # process uncertainty
    fk.Q = np.array([[0.01, 0, 0, 0],
                     [0, 0.01, 0, 0],
                     [0, 0, 0.01, 0],
                     [0, 0, 0, 0.01]])
    # initial state (location and velocity)
    fk.x[0:2] = np.array([[input_x[0], input_y[0]]]).reshape(2, 1)

    ## zs应该是n行2列的矩阵
    # zs = np.array([[x, y]]).reshape(-1,2)
    zs = np.column_stack([input_x,input_y])

    mu, cov, _, _ = fk.batch_filter(zs)
    M, P, _, _ = fk.rts_smoother(mu, cov)
    mus = [x[0] for x in mu]
    return M[:,0],M[:,1]

if __name__=='__main__':
    txt_path='../reconstruction_txts/exp0104_reconstructed.txt'
    data = np.loadtxt(txt_path, dtype=str, delimiter=',')
    new_data = data.astype(np.float64)
    filename=os.path.basename(txt_path)

    id_100_data = new_data[new_data[:, 1] == 100, :]
    input_x,input_y = id_100_data[:,2],id_100_data[:,3]

    smoothed_x,smoothed_y= trajectory_smoothing_by_RTS(input_x,input_y)

    p1, = plt.plot(input_x,input_y,'cyan', alpha=0.5)
    p2, = plt.plot(smoothed_x,smoothed_y,c='b')
    # p3, = plt.plot(mus,c='r')
    # p4, = plt.plot([0, len(zs)], [0, len(zs)], 'g') # perfect result
    plt.legend([p1, p2],["measurement", "RKS"], loc=3)
    plt.show()


