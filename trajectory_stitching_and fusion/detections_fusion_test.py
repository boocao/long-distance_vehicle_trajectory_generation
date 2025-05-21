import os
import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy import signal
from collections import Counter
from multiprocessing import Pool,cpu_count,Process
# from draw_utils import draw_points

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

            bbox_iou=calculate_IOU(bbox1, bbox2)
            print(bbox_iou)

            if bbox_iou >0.6:
                print('+1')
                # per_frame_data[i, 2] = (per_frame_data[i, 2] + per_frame_data[j, 2])/2.
                # per_frame_data[i, 3] = (per_frame_data[i, 3] + per_frame_data[j, 3])/2.
                # per_frame_data[i, 4] = (per_frame_data[i, 4] + per_frame_data[j, 4])/2.
                # per_frame_data[i, 5] = (per_frame_data[i, 5] + per_frame_data[j, 5])/2.
                X1=3416
                x_max = X1 + 20
                x_min = X1 - 20
                medium_x = (per_frame_data[i, -2] + per_frame_data[j, -2]) / 2.
                weight1 = (medium_x - x_min) / (x_max - x_min)
                weight2 = (x_max - medium_x) / (x_max - x_min)
                if per_frame_data[i, -3]==1:
                    per_frame_data[i, 2] = (per_frame_data[i, 2]*weight2 + per_frame_data[j, 2]*(weight1))
                    per_frame_data[i, 3] = (per_frame_data[i, 3]*weight2 + per_frame_data[j, 3]*(weight1))
                    per_frame_data[i, 4] = (per_frame_data[i, 4]*weight2 + per_frame_data[j, 4]*(weight1))
                    per_frame_data[i, 5] = (per_frame_data[i, 5]*weight2 + per_frame_data[j, 5]*(weight1))
                elif per_frame_data[i, -3]==2:
                    per_frame_data[i, 2] = (per_frame_data[i, 2]*weight1 + per_frame_data[j, 2]*(weight2))
                    per_frame_data[i, 3] = (per_frame_data[i, 3]*weight1 + per_frame_data[j, 3]*(weight2))
                    per_frame_data[i, 4] = (per_frame_data[i, 4]*weight1 + per_frame_data[j, 4]*(weight2))
                    per_frame_data[i, 5] = (per_frame_data[i, 5]*weight1 + per_frame_data[j, 5]*(weight2))
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
    path='E:/07Experiment-data/VideoSets/02AerialVideos-05LongTextFiles/exp0502/'
    det1_path = path+'exp4/output.txt'
    det2_path = path+'exp5/output.txt'

    data1 = np.loadtxt(det1_path, delimiter=',',dtype=bytes).astype(str)
    data1 = data1.astype(np.float64)
    data2 = np.loadtxt(det2_path, delimiter=',',dtype=bytes).astype(str)
    data2 = data2.astype(np.float64)

    ## 此时数据格式为mot格式：[x1,y1,w,h]
    remaining_data1=data1[data1[:, 0] >=1, :] ## 帧数从1开始
    remaining_data2=data2[data2[:, 0] >=1, :] ## 帧数从1开始
    X1,Y1,delta_pixel=3416,1109,200  ## 拼接点X1,Y1=3840-252,728;和右边视频留存的像素量delta=400
    remaining_data2[:,2]=remaining_data2[:,2]+X1-delta_pixel

    ##合并后融合单帧iou大于0.6的dets
    temp1=np.full([remaining_data1.shape[0],3], 1, dtype=float)
    remaining_data1=np.concatenate([remaining_data1,temp1],axis=1)
    temp2=np.full([remaining_data2.shape[0],3], 2, dtype=float)
    remaining_data2=np.concatenate([remaining_data2,temp2],axis=1)
    data_fusion=np.concatenate([remaining_data1,remaining_data2],axis=0)
    data_fusion[:,4] = (data_fusion[:,2] + data_fusion[:,4])
    data_fusion[:,5] = (data_fusion[:,3] + data_fusion[:,5])
    data_fusion[:,-2] = (data_fusion[:,2] + data_fusion[:,4])/2.
    data_fusion[:,-1] = (data_fusion[:,3] + data_fusion[:,5])/2.

    data_fusion1=data_fusion[data_fusion[:,-3]==1,:]
    data_fusion2=data_fusion[data_fusion[:,-3]==2,:]
    # x_max=data_fusion1[:,-2].max()-100
    # x_min=data_fusion2[:,-2].min()+100
    data_fusion1=data_fusion1[data_fusion1[:,-2]<=X1+20,:] ##设置20没问题
    data_fusion2=data_fusion2[data_fusion2[:,-2]>=X1-20,:]
    data_fusion=np.concatenate([data_fusion1,data_fusion2],axis=0)
    data_fusion=data_fusion[data_fusion[:, 0] <= 22891, :]

    time1 = time.time()
    task_list1 = list()
    for i in range(int(data_fusion[:, 0].max())):
    # for i in range(int(1000)):
        per_frame_data = data_fusion[data_fusion[:, 0] == (i + 1), :]
        # per_frame_data = per_frame_data[np.argsort(per_frame_data[:, 2]), :]
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

    new_data=new_data_fusion[:,:11]
    save_path = path+ 'det_test.txt'
    np.savetxt(save_path, new_data, fmt="%d,%d,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d,%d", delimiter="\n")