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
    ####获取所要数据
    data_df1 = pd.read_csv('./data/truck0003.csv')
    data_df2 = pd.read_csv('./data/rts_exp0607.csv')
    ## 补充时间差 RTS
    # data_df2['time']=data_df2['time']+53.09167467      ## 65656.63秒与65603.53832533333秒 差值=53.09167467秒
    data_df2['time']=data_df2['time']+52.35187     ## 66038.969秒与65986.617125秒 差值=52.35187秒
    ##unsmoothed
    # data_df2['time']=data_df2['time']+53.1250080001     ## 65656.63秒与65603.5049919999秒 差值=53.1250080001秒
    # data_df2['time']=data_df2['time']+52.4185416667     ## 66038.969秒与65986.5504583333秒 差值=52.4185416667秒

    #### 坐标暂时用不了
    # ## 像素m转换gps，1920,1080=164.57142857,92.57142857  30.425829,114.108554
    # ## 最近点 1922.4995,1001.7045=164.4402,85.6804,12.5737  30.4264778,114.10033600000001
    # data2=data_df2.values
    # data2=data2[data2[:,1]==3,:]
    # data2[:, 2]= data2[:, 2]-1920
    # data2[:, 3]= data2[:, 3]-1080
    # data2[:, 2]= data2[:, 2]*((114.10033600000001-114.108554)/(1922.4995-1920))
    # data2[:, 3]= data2[:, 3]*((30.4264778-30.425829)/(1001.7045-1080))
    # data2[:, 2] = data2[:, 2]+114.10855
    # data2[:, 3] = data2[:, 3]+30.425829
    # plt.figure(figsize=(10, 4))
    # plt.plot(data1[:,3], data1[:,2])
    # plt.plot(data2[:,2], data2[:,3])
    # plt.show()
    ####################################################################################################################
    data_df1=data_df1.fillna(-1)
    data1=data_df1.values
    data1=data1[data1[:,5]!=-1,:] #4,5,6为速度
    # data_velocity=np.sqrt(data1[:,5]**2+data1[:,6]**2)
    array_time_velocity1=np.column_stack((data1[:,0],data1[:,4:7]))
    # print(array_time_velocity1)

    data2=data_df2.values
    data2=data2[data2[:,2]>=175,:]
    data2=data2[data2[:,2]<=(3840-175),:]
    data2=data2[data2[:,3]>=100,:]
    data2=data2[data2[:,3]<=(2160-100),:]
    # data2=data2[data2[:,1]==5,:]
    array_time_velocity2=np.column_stack((data2[:,:2],data2[:,6:10],data2[:,12:18])) ##增加要用到的数据

    ##对其时间
    array_time_velocity1[:,0]=np.round(array_time_velocity1[:,0],2)
    array_time_velocity2[:,0]=np.round(array_time_velocity2[:,0],2)
    print(array_time_velocity1[:,0].shape)
    print(array_time_velocity2[:,0].shape)

    new_array_time_velocity=np.empty((0,15))
    for i in array_time_velocity2[:,0]:
        if i in array_time_velocity1[:,0]:
            print(i)
            temp1=(array_time_velocity1[array_time_velocity1[:,0]==i,:])
            temp2=(array_time_velocity2[array_time_velocity2[:,0]==i,:])
            array_time_velocity_i=np.column_stack((temp2[:,:],temp1[:,1:]))
            new_array_time_velocity=np.row_stack((new_array_time_velocity,array_time_velocity_i))

    plt.figure(figsize=(10, 4))
    plt.plot(new_array_time_velocity[:,0], new_array_time_velocity[:,8])
    plt.plot(new_array_time_velocity[:,0], new_array_time_velocity[:,12])
    plt.title("Trajectory")
    plt.ylabel("Y(velocity)")
    plt.xlabel("X(time)")
    plt.show()

    dict_temp={'time':new_array_time_velocity[:,0],'id':new_array_time_velocity[:,1],
                'x':new_array_time_velocity[:,2],'y':new_array_time_velocity[:,3],
                'w': new_array_time_velocity[:, 4], 'h': new_array_time_velocity[:, 5],
               'vx': new_array_time_velocity[:, 6], 'vy': new_array_time_velocity[:, 7],
               'v': new_array_time_velocity[:, 8], 'ax': new_array_time_velocity[:, 9],
               'ay': new_array_time_velocity[:, 10],'a': new_array_time_velocity[:, 11],
               'Speed2D-(m/s)': new_array_time_velocity[:, 12],
               'VelForward-(m/s)': new_array_time_velocity[:, 13],
               'VelLateral-(m/s)': new_array_time_velocity[:, 14],
                }
    data_frame=pd.DataFrame(dict_temp)
    data_frame.to_csv('./csvs/rts_exp0607_truck0003_velocity.csv', index=False)
    ####################################################################################################################
    data_df1=data_df1.fillna(-1)
    data1=data_df1.values
    data1=data1[data1[:,7]!=-1,:] #7,8,9,10为加速度，使用水平面的数据
    data_acceleration=np.sqrt(data1[:,7]**2+data1[:,8]**2)
    array_time_acceleration1=np.column_stack((data1[:,0],data_acceleration,data1[:,7:9]))

    data2=data_df2.values
    data2=data2[data2[:,2]>=175,:]
    data2=data2[data2[:,2]<=(3840-175),:]
    data2=data2[data2[:,3]>=100,:]
    data2=data2[data2[:,3]<=(2160-100),:]
    # data2=data2[data2[:,1]==5,:]
    array_time_acceleration2=np.column_stack((data2[:,:2],data2[:,6:10],data2[:,12:18])) ##增加要用到的数据

    ##对其时间
    # array_time_velocity1=array_time_velocity1[array_time_velocity1[:,0]>=65643,:]
    # array_time_velocity1=array_time_velocity1[array_time_velocity1[:,0]<=65671,:]
    array_time_acceleration1[:,0]=np.round(array_time_acceleration1[:,0],2)
    array_time_acceleration2[:,0]=np.round(array_time_acceleration2[:,0],2)
    print(array_time_acceleration1[:,0].shape)
    print(array_time_acceleration2[:,0].shape)
    print(array_time_acceleration1[array_time_acceleration1[:,0]==66304.04,:])

    array_time_acceleration=np.empty((0,15))
    for i in array_time_acceleration2[:,0]:
        if i in array_time_acceleration1[:,0]:
            print(i)
            temp1=(array_time_acceleration1[array_time_acceleration1[:,0]==i,:])
            if temp1.shape[0]>1:
                temp=np.mean(temp1,axis=0)
                temp1=temp.reshape((1,-1))
            temp2=(array_time_acceleration2[array_time_acceleration2[:,0]==i,:])
            array_time_acceleration_i=np.column_stack((temp2[:,:],temp1[:,1:]))
            array_time_acceleration=np.row_stack((array_time_acceleration,array_time_acceleration_i))

    plt.figure(figsize=(10, 4))
    plt.plot(array_time_acceleration[:,0], array_time_acceleration[:,11])
    plt.plot(array_time_acceleration[:,0], array_time_acceleration[:,12])
    plt.title("Trajectory")
    plt.ylabel("Y(Acceleration)")
    plt.xlabel("X(Time)")
    plt.show()

    dict_temp={'time':array_time_acceleration[:,0],'id':array_time_acceleration[:,1],
                'x':array_time_acceleration[:,2],'y':array_time_acceleration[:,3],
                'w': array_time_acceleration[:, 4], 'h': array_time_acceleration[:, 5],
               'vx': array_time_acceleration[:, 6], 'vy': array_time_acceleration[:, 7],
               'v': array_time_acceleration[:, 8],'ax': array_time_acceleration[:, 9],
               'ay': array_time_acceleration[:, 10],'a': array_time_acceleration[:, 11],
               'Accel-': array_time_acceleration[:, 12],
               'AccelForward-(m/s_)': array_time_acceleration[:, 13],
               'AccelLateral-(m/s_)': array_time_acceleration[:, 14],
                }
    data_frame=pd.DataFrame(dict_temp)
    data_frame.to_csv('./csvs/rts_exp0607_truck0003_acceleration.csv', index=False)


