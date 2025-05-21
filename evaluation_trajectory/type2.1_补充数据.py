import matplotlib
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from PIL import Image
import os
from scipy import signal
from matplotlib.font_manager import FontProperties


if __name__ == "__main__":
    dataframe=pd.read_csv('./CitySIM/FreewayC-01.csv',header=0)
    dataframe = dataframe[['frameNum', 'carId', 'carCenterXft', 'carCenterYft', 'speed', 'laneId','headXft','headYft','tailXft','tailYft']]
    data = dataframe.values

    ##补充长度数据
    length_data=np.sqrt((data[:,6]-data[:,8])**2+(data[:,7]-data[:,9])**2)
    data = np.column_stack((data[:, :6], length_data))

    ##补充速度、加速度数据
    new_data = np.empty((0, 8))
    for i in range(int(data[:, 1].max())):  # 遍历每个车辆ID
        per_id_data = data[data[:, 1] == (i + 1), :]  # 获取特定车辆的数据
        per_id_data = per_id_data[np.argsort(per_id_data[:, 0]), :]  # 按时间帧排序
        if per_id_data.shape[0] <= 1:  # 如果该车辆数据少于2个点，无法计算加速度
            continue
        delta_speed = np.diff(per_id_data[:, 4])
        delta_time = np.diff(per_id_data[:, 0])  # 假设帧间隔为1，如果有实际时间数据，可以替换
        acc = delta_speed / delta_time  # 计算加速度
        new_per_id_data = np.column_stack((per_id_data[1:,:5], acc,per_id_data[1:,5:]))
        new_data = np.row_stack((new_data, new_per_id_data))

    ##补充前车、后车ID数据
    nnew_data = np.empty((0, 10))
    for i in range(int(new_data[:, 0].max())+1):
        per_frame_data = new_data[new_data[:, 0] == (i), :]  ## citysim从0帧开始
        if per_frame_data.shape[0] == 0:
            continue
        for j in np.unique(per_frame_data[:, 6]):
            per_lane_data = per_frame_data[per_frame_data[:, 6] == (j), :]  ##citysim从车道0开始
            if per_lane_data.shape[0] == 0:
                continue
            ## 这里用车道号来判断车辆流向，没有用速度方向来判断
            if (j) <= 3:
                per_lane_data = per_lane_data[np.argsort(-per_lane_data[:, 2]), :]
            else:
                per_lane_data = per_lane_data[np.argsort(per_lane_data[:, 2]), :]
            per_lane_id_array = per_lane_data[:, 1]
            per_lane_id_numbers = per_lane_id_array.shape[0]

            for index, k in enumerate(per_lane_id_array):
                if per_lane_id_numbers == 1:
                    per_id_data = per_lane_data[per_lane_data[:, 1] == k, :]
                    preceding_id = 0
                    following_id = 0
                    new_per_id_data = np.column_stack((per_id_data, preceding_id, following_id))
                else:
                    per_id_data = per_lane_data[per_lane_data[:, 1] == k, :]
                    if index == 0:
                        preceding_id = per_lane_id_array[index + 1]
                        following_id = 0
                    elif index == per_lane_id_numbers - 1:
                        preceding_id = 0
                        following_id = per_lane_id_array[index - 1]
                    else:
                        preceding_id = per_lane_id_array[index + 1]
                        following_id = per_lane_id_array[index - 1]
                    new_per_id_data = np.column_stack((per_id_data, preceding_id, following_id))
                nnew_data = np.row_stack((nnew_data, new_per_id_data))

    ## 单位转换
    nnew_data[:,2:4]=nnew_data[:,2:4]*0.3048
    nnew_data[:,4]=nnew_data[:,4]*(5280/3600)*0.3048
    nnew_data[:,5]=nnew_data[:,5]*(5280/3600)*0.3048*30
    nnew_data[:,7]=nnew_data[:,7]*0.3048
    dataframe = pd.DataFrame(data=nnew_data,
                             columns=['frameNum', 'carId', 'carCenterXm', 'carCenterYm', 'speed', 'acceleration',
                                      'laneId','length', 'preceding_id', 'following_id'])
    dataframe.to_csv('./CitySIM/FreewayC01_for_snp.csv', index=False)
    ############################################################################
    ##计算一致性所需的数据
    dataframe=pd.read_csv('./CitySIM/FreewayC01_for_snp.csv',header=0)
    data = dataframe.values
    ## 计算轨迹排队一致性,需要剔除前面一两个数据
    snp_data=np.empty((0, 13))
    for frame in range(int(data[:,0].max())):
        current_frame = (frame)
        per_frame_data = data[data[:, 0] == current_frame, :]
        if per_frame_data.shape[0] == 0:
            continue
        for id in per_frame_data[:, 1]:
            per_frame_id_data = per_frame_data[per_frame_data[:, 1] == id, :]
            current_id=id
            preceding_id=per_frame_id_data[:,8]
            if preceding_id==0:
                continue

            ##分别提取每个车辆对的本车数据和前车数据
            current_id_data = data[data[:, 1] == current_id, :]
            current_id_data=current_id_data[current_id_data[:, 0] == current_frame, :]
            preceding_id_data = data[data[:, 1] == preceding_id, :]
            preceding_id_data = preceding_id_data[preceding_id_data[:, 0] == current_frame, :]

            ####计算距离-速度的排队一致性
            delta_v=current_id_data[0,4]-preceding_id_data[0,4]
            delta_a=current_id_data[0,5]-preceding_id_data[0,5]
            obs_snp=np.sqrt((current_id_data[0,2]-preceding_id_data[0,2])**2+(current_id_data[0,3]-preceding_id_data[0,3])**2)
            # ttc=(obs_snp-current_id_data[0,7]/2-preceding_id_data[0,7]/2)/(delta_v)
            new_per_frame_id_data= np.column_stack((per_frame_id_data,delta_v,delta_a,obs_snp))
            snp_data = np.row_stack((snp_data, new_per_frame_id_data))

    snp_data = pd.DataFrame(data=snp_data,
                             columns=['frameNum', 'carId', 'carCenterXm', 'carCenterYm', 'speed', 'acceleration',
                                      'laneId','length', 'preceding_id', 'following_id','delta_v','delta_a',
                                      'obs_snp'])
    snp_data.to_csv('./CitySIM/FreewayC01_snp_data.csv', index=False)


