import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from draw_utils import draw_points,draw_xy
from scipy import signal
from collections import Counter


if __name__=='__main__':
    filepath = "./reconstruction_txts/exp0607_reconstructed.txt"
    filename=os.path.basename(filepath)

    data = np.loadtxt(filepath,delimiter=',',dtype=bytes).astype(str)
    data = data.astype(np.float64)

    ## 更改车辆类别 ['car', 'truck', 'bus']
    id_array_need_to_car=[  ]
    id_array_need_to_truck=[
        95, 96, 97, 98,

    ]
    id_array_need_to_bus=[ ]
    ## 统一同一车辆ID数据
    id_array_need_to_change=[

                             ]
    ###################################
    Data = np.empty((0, data.shape[1]))
    for j in range(int(data[:, 1].max())):
        per_id_data = data[data[:, 1] == (j+1), :]
        if (j+1) in [ ]:
            continue
        if per_id_data.shape[0] <= 0:
            continue
        for i in id_array_need_to_car:
            if per_id_data[0,1]==i:
                per_id_data[:,8]=0
        for j in id_array_need_to_truck:
            if per_id_data[0,1]==j:
                per_id_data[:,8]=1
        for k in id_array_need_to_bus:
            if per_id_data[0,1]==k:
                per_id_data[:,8]=2

        for m in id_array_need_to_change:
            for n in m:
                if per_id_data[0,1]==n:
                    per_id_data[:,1]=m[0]
        Data = np.row_stack((Data, per_id_data))

    save_path = './reconstruction_txts/' + filename.split('.')[0] + '_changed.txt'
    np.savetxt(save_path, Data,
               fmt=['%0.0f', '%0.0f', '%0.4f', '%0.4f', '%0.4f', '%0.4f', # frame, id, centers_x, centers_y, width, height, 0-5
                    '%0.4f', '%0.4f', '%0.0f'  # denoised_x, denoised_y, 6-7, class, 8,共9个数据
                    ], delimiter=',')