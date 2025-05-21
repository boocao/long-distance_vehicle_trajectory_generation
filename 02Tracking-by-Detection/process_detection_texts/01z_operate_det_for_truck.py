import os
import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy import signal
from collections import Counter
from multiprocessing import Pool,cpu_count,Process

def calculate_IOU(bbox1,bbox2):
    """
    坐标格式是（左，上，右，下）
    box1:(x11,y11,x12,y12)
    box2:(x21,y21,x22,y22)
    """
    # x11,y11,x12,y12 = bbox1
    # x21,y21,x22,y22 = bbox2
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2
    width1 =  np.maximum(0,xmax1-xmin1)
    height1 = np.maximum(0,ymax1-ymin1)
    width2  = np.maximum(0,xmax2-xmin2)
    height2 = np.maximum(0,ymax2-ymin2)
    area1 = width1*height1
    area2 = width2*height2
    #计算交集，需要计算交集部分的左、上、右、下坐标
    xi1 = np.maximum(xmin1,xmin2)
    yi1 = np.maximum(ymin1,ymin2)
    xi2 = np.minimum(xmax1,xmax2)
    yi2 = np.minimum(ymax1,ymax2)
    #计算交集部分面积
    w = np.maximum(0,xi2-xi1)
    h = np.maximum(0,yi2-yi1)
    intersection = w*h
    #计算并集,计算iou
    union = area1+area2-intersection
    iou = intersection/union
    return iou

def new_per_frame_data_iou_matched(per_frame_data):
    new_per_frame_data= np.empty((0, per_frame_data.shape[1]))
    need_remove_data_list=list()
    for i in range(len(per_frame_data[:, 0])):
        for j in range(len(per_frame_data[:, 0])):
            if j<=i:
                continue
            ##计算bbox_iou, 坐标格式是[x1,y1,x2,y2]
            bbox1_array=np.empty((1,4))
            bbox2_array=np.empty((1,4))
            bbox1_array[:]=per_frame_data[i, 2:6]
            bbox2_array[:]=per_frame_data[j, 2:6]
            bbox1=bbox1_array.flatten()
            bbox2=bbox2_array.flatten()
            # bbox1[0]=bbox1[0]-10
            # bbox1[2]=bbox1[2]+10
            # bbox2[0]=bbox1[0]-10
            # bbox2[2]=bbox1[2]+10

            bbox_iou=calculate_IOU(bbox1, bbox2)
            print(bbox_iou)

            if bbox_iou >=0.7:
                print('+1')
                per_frame_data[i, 2] = (per_frame_data[i, 2] + per_frame_data[j, 2]) / 2.
                per_frame_data[i, 3] = (per_frame_data[i, 3] + per_frame_data[j, 3]) / 2.
                per_frame_data[i, 4] = (per_frame_data[i, 4] + per_frame_data[j, 4]) / 2.
                per_frame_data[i, 5] = (per_frame_data[i, 5] + per_frame_data[j, 5]) / 2.
                need_remove_data_list.append(j)
        new_per_frame_data = np.row_stack((new_per_frame_data, per_frame_data[i, :]))

    print(new_per_frame_data.shape)
    remaining_per_frame_data= np.empty((0, new_per_frame_data.shape[1]))
    for k in range(len(new_per_frame_data[:, 0])):
        if k in need_remove_data_list:
            print('++')
            continue
        remaining_per_frame_data = np.row_stack((remaining_per_frame_data, new_per_frame_data[k, :]))
    print(remaining_per_frame_data.shape)
    return remaining_per_frame_data

if __name__=='__main__':
    root_path = 'E:/07Experiment-data/VideoSets/02AerialVideos-05LongTextFiles/exp0502/exp5/'
    # root_path = 'E:/07Experiment-data/VideoSets/02AerialVideos-05LongTextFiles/exp0501/exp_left-1/'

    det_path = root_path + '/det.txt'
    data = np.loadtxt(det_path, delimiter=',',dtype=bytes ).astype(str)
    data = data.astype(np.float64)

    data_fusion = data[np.argsort(data[:, 0]), :]

    time1 = time.time()
    task_list1 = list()
    for i in range(int(data_fusion[:, 0].max())):
    # for i in range(int(3000)):
        per_frame_data = data_fusion[data_fusion[:, 0] == (i + 1), :]
        if per_frame_data.shape[0]<=0:
            continue
        task_list1.append(per_frame_data)

    pool = Pool(processes=cpu_count()-6)
    new_per_frame_data_iou_matched_list = pool.map(new_per_frame_data_iou_matched, task_list1)
    pool.close()
    pool.join()

    new_data_fusion = np.empty((0, data_fusion.shape[1]))
    for new_per_frame_data_iou_matched in new_per_frame_data_iou_matched_list:
        new_data_fusion = np.row_stack((new_data_fusion, new_per_frame_data_iou_matched))
    time2 = time.time()
    print(str(time2 - time1) + 's')

    save_path = root_path + '/det_.txt'
    np.savetxt(save_path, new_data_fusion, fmt="%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d,%d", delimiter="\n")