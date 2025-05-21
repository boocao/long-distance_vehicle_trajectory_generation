import os
import re
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# 导入字体属性相关的包或类
from matplotlib.font_manager import FontProperties

if __name__=='__main__':
    # ## 1.1合并adas小车数据
    # path='F:/11TrajectoryData/TrajectoryDataVerification/VerificationData/小车数据/'
    # merge_data=np.empty((0,5))
    # for d in os.listdir(path):
    #     if os.path.isdir(os.path.join(path, d)):
    #         txt_path = os.path.join(path, d)
    #         for f in os.listdir(txt_path):
    #             file = pd.read_csv(os.path.join(txt_path, f), header=None, sep='\s+')
    #             file_array=file.values
    #             car_id = int(re.findall(r'\d+',d)[0])
    #             temp_ = np.full((file_array.shape[0], 1), car_id)
    #             every_txt_data=np.column_stack((file_array[:,1],temp_,file_array[:,2:5]))
    #             merge_data=np.row_stack((merge_data,every_txt_data))
    # merge_data=merge_data[merge_data[:,2]!=0,:]
    # merge_data=merge_data[merge_data[:,3]!=0,:]
    #
    # output_path='./merge_car_data_all_time.txt'
    # np.savetxt(output_path, merge_data, fmt=['%0.0f', '%0.0f', '%0.8f', '%0.8f', '%0.8f'],delimiter=',')
    #
    # ## 1.2时分秒转秒
    # dataframe = pd.read_csv('merge_car_data_all_time.txt', header=None)
    # data=dataframe.values
    # for i in range(len((data))):
    #     time_str=str(int(data[i,0]))
    #     # print(time_str)
    #     time_second=int(time_str[8:10])*3600+int(time_str[10:12])*60+int(time_str[12:14])+int(time_str[14:17])*0.001
    #     print(time_second)
    #     data[i,0]=time_second
    # new_dataframe=pd.DataFrame(data)
    # new_dataframe.to_csv('new_merge_car_data_all_time.csv', index=False, header=False, encoding='utf-8')

    # ##绘制轨迹图
    # data_df = pd.read_csv('merge_car_data_all_time.csv', header=None)
    # data=data_df.values
    # colours = np.random.rand(32, 3)
    # plt.figure(figsize=(10, 4))
    # for i in np.unique(data[:,1]):
    #     print(i)
    #     id_data=data[data[:,1]==int(i),:]
    #     id_x=id_data[:,2]
    #     id_y=id_data[:,3]
    #     # plt.scatter(id_x, id_y, color=colours[int(i) % 32, :],s=1, linewidths=1)
    #     plt.plot(id_x, id_y, color=colours[int(i) % 32, :])
    # plt.title("Trajectory")
    # plt.ylabel("Y(m)")
    # plt.xlabel("X(m)")
    # plt.legend(loc="upper left")
    # plt.show()

    # ### 绘制时空图
    # data_df = pd.read_csv('merge_car_data_all_time.csv', header=None)
    # data = data_df.values
    #
    # plt.figure(figsize=(10, 4))
    # font = FontProperties(fname=r"C:WindowsFontstimes.TTF", size=10)
    # for i in np.unique(data[:,1]):
    #     # 循环绘制轨迹图
    #     print(i)
    #     cardata = data[data[:,1] == int(i),:]
    #     cardata = cardata[np.argsort(cardata[:, 0]),:]
    #
    #     time = cardata[:,0]
    #     # min_x=cardata[:,2].min()
    #     # min_y=cardata[:,2].max()
    #     # dist = np.sqrt(np.square(cardata[:,2]-min_x) + np.square(cardata[:,3]-min_y))
    #     dist = np.sqrt(np.square(cardata[:,3]))
    #     # 将速度赋值给变量 v，同时定义速度为颜色映射
    #     v = cardata[:,4]
    #     #设定每个图的colormap和colorbar所表示范围是一样的，即归一化
    #     norm = matplotlib.colors.Normalize(vmin=0, vmax=25)
    #     ax = plt.scatter(time,dist, marker = '.', s=1, c=v, cmap='jet_r', norm = norm)
    #     # ax1 = plt.plot(time,dist, marker = '.', c=v)
    #     i = i + 1
    #     # 添加颜色条
    # plt.clim(0, 25)
    # plt.colorbar()
    #
    # # 设定每个图的colormap和colorbar所表示范围是一样的，即归一化
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=25)
    # # 设置 X 坐标轴刻度
    # # plt.text(x, y, '8:05', fontproperties=font)
    # # plt.text(x, y, '8:10', fontproperties=font)
    # # plt.text(x, y, '8:15', fontproperties=font)
    # # plt.text(x, y, '8:20', fontproperties=font)
    # plt.ylabel("Dist")
    # plt.xlabel("Time")
    # plt.show()
    ####################################################################################################################
    ####################################################################################################################
    # ## 2.1大车数据(不用合并)
    # path='F:/11TrajectoryData/TrajectoryDataVerification/VerificationData/data_truck/'
    # filelist=[]
    # for d in os.listdir(path):
    #     if os.path.isfile(os.path.join(path, d)):
    #         txt_path = os.path.join(path, d)
    #         print(txt_path)
    #         file=pd.read_csv(txt_path,header=0,dtype=str) ##str可以保存数据精度，后面可以astype('int64')
    #         print(file)
    #         filelist.append(file)
    # file_concat=pd.concat(filelist,axis=0)
    # file_concat.to_csv('merge_truck_data_all_time.csv',index=False)

    ## 2.2时间转秒
    # dataframe = pd.read_csv(r'./VerificationData/data_truck_ins/weituo20220805_0002.csv',header=0,dtype=str)
    # data=dataframe.values
    # for i in range(len((data))):
    #     time_str=data[i,0]
    #     print(time_str)
    #     # print(dataframe['Time (-)'][i])
    #     time_str_=time_str.split(' ')[1]
    #     hour_str=time_str_.split(':')[0]
    #     munitue_str=time_str_.split(':')[1]
    #     second_str=time_str_.split(':')[2].split('.')[0]
    #     msceond_str=time_str_.split('.')[1]
    #     time_second=int(hour_str)*3600+int(munitue_str)*60+int(second_str)+int(msceond_str)*0.0001
    #     print(time_second)
    #     dataframe['Time (-)'][i]=str(time_second)
    # id_col=np.full((data.shape[0], 1), 2).flatten()
    # dict_temp={'ID':id_col}
    # data_frame2=pd.DataFrame(dict_temp)
    # new_dataframe=pd.concat([dataframe,data_frame2],axis=1)
    # new_dataframe=new_dataframe[['Time (-)','ID','PosLat -  (_)','PosLon -  (_)','Speed2D -  (m/s)',
    #                              'VelForward -  (m/s)','VelLateral -  (m/s)',
    #                              'AccelX -  (m/s_)','AccelY -  (m/s_)',
    #                              'AccelForward -  (m/s_)','AccelLateral -  (m/s_)']]
    # new_dataframe.to_csv('truck0002.csv', index=False)

    # ### 绘制时空图
    # data_df = pd.read_csv('truck0002.csv')
    # # time_temp=65538  ## 应该是快了53秒   ##0003为65992
    # # data_df=data_df[data_df['Time (-)']>=time_temp]
    # # data_df=data_df[data_df['Time (-)']<=(time_temp+170)]  ## 0003为190
    # data_df=data_df.fillna(-1)
    # data=data_df.values
    # data=data[data[:,3]!=-1,:]
    # colours = np.random.rand(32, 3)
    # # plt.figure(figsize=(10, 4))
    # for i in np.unique(data[:,1]):
    #     print(i)
    #     id_data=data[data[:,1]==int(i),:]
    #     id_x=id_data[:,3]
    #     id_y=id_data[:,2]
    #     # plt.scatter(id_x, id_y, color=colours[int(i) % 32, :],s=1, linewidths=1)
    #     plt.plot(id_x, id_y, color=colours[int(i) % 32, :])
    #     # plt.scatter(114.100339,30.426458)
    # plt.title("Trajectory")
    # plt.ylabel("Y(m/s)")
    # plt.xlabel("X(s)")
    # plt.legend(loc="upper left")
    # plt.show()

    # 3.1 视频轨迹数据时间转秒
    dataframe = pd.read_csv('./VideoTrajectory/exp0606_dataset_rts.csv')
    # print(dataframe)
    data=dataframe.values
    dataframe['frame']=dataframe['frame'].astype(float)
    for i in range(len((data))):
        frame_str=data[i,0]
        print(frame_str)
        start_second = 18*3600+13*60+7+0.104992  ##0606
        # start_second = 18*3600+19*60+17+0.117125 ##0607
        time_second=int(frame_str)*(1.0/30)
        dataframe['frame'][i]=start_second+time_second
    new_dataframe= dataframe.rename(columns={'frame':'time'})
    new_dataframe=new_dataframe[new_dataframe['id']!=1] ##去掉1,4和2,4
    new_dataframe=new_dataframe[new_dataframe['id']!=4]
    new_dataframe.to_csv('./data/rts_exp0606.csv', index=False)
