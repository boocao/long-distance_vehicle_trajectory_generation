import os
import re
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

import matplotlib
import matplotlib.pyplot as plt
# 导入字体属性相关的包或类
from matplotlib.font_manager import FontProperties

if __name__=='__main__':
    # # 计算时间差
    # data_df = pd.read_csv('./csvs/truck0002.csv')
    # time_temp=65538  ## 应该是快了53秒   ##0003为65992
    # data_df=data_df[data_df['Time (-)']>=time_temp]
    # data_df=data_df[data_df['Time (-)']<=(time_temp+170)]  ## 0003为190
    # data_df=data_df.reset_index(drop=True)
    #
    # ##总结为常用结构,注意预先重置索引
    # data=data_df.values
    # distance_data=np.zeros((data.shape[0],2))
    # for i in range(len(data[:,0])):
    #     # print(i)
    #     if np.isnan(data[i,2]):
    #         continue
    #     else:
    #         distance = ((data[i, 3] - 114.100339) ** 2 + (data[i, 2] - 30.426458) ** 2)        ## 0002
    #         # distance = ((data[i, 3] - 114.108554) ** 2 + (data[i, 2] - 30.425829) ** 2)       ## 0003
    #     distance_data[i,0]=distance
    # dict_temp={'distance1':distance_data[:,0],'distance2':distance_data[:,1]}
    # data_frame2=pd.DataFrame(dict_temp)
    # new_dataframe=pd.concat([data_df,data_frame2],axis=1)
    #
    # new_dataframe=new_dataframe.sort_values(by='distance1')
    # new_dataframe=new_dataframe[new_dataframe['distance1']!=0]
    # new_dataframe.to_csv('./csvs/truck0002_temp.csv', index=False)

    data_df = pd.read_csv('./data/rts_exp0607.csv')
    data_df=data_df[['time','id','center_x_pixel','center_y_pixel','width_pixel','height_pixel','center_x','center_y','width','height','class','lane',
                     'calculated_vx','calculated_vy','calculated_v','calculated_ax','calculated_ay','calculated_a']]
    data_df=data_df[data_df['id']==3]

    data_df=data_df.reset_index(drop=True)
    data=data_df.values
    distance_data=np.zeros((data.shape[0],2))
    for i in range(len(data[:,0])):
        print(i)
        distance=((data[i,2]-1920)**2+(data[i,3]-1080)**2)
        print(distance)
        distance_data[i,0]=distance
    dict_temp={'distance1':distance_data[:,0],'distance2':distance_data[:,1]}
    data_frame2=pd.DataFrame(dict_temp)
    data_frame2=data_frame2.reset_index(drop=True)
    new_dataframe=pd.concat([data_df,data_frame2],axis=1)

    new_dataframe=new_dataframe.sort_values(by='distance1')
    # print(new_dataframe['distance1'])
    new_dataframe.to_csv('./data/rts_exp0607_temp.csv', index=False)