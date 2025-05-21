import matplotlib
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ## ['frameNum', 'carId', 'carCenterXm', 'carCenterYm', 'speed', 'acceleration',
    ##  'laneId', 'length', 'preceding_id', 'following_id',
    ## 'delta_v', 'delta_a', 'obs_snp'] #10,11,12
    dataframe=pd.read_csv('./CitySIM/FreewayC01_snp_data.csv',header=0)
    grouped_dataframe = dataframe.groupby('carId')
    selected_data=np.empty((0, dataframe.shape[1]))
    ####1.提取数据-方式1
    for id, id_dataframe in grouped_dataframe:
        id_dataframe = id_dataframe.sort_values('frameNum')  # 按帧号排序
        id_data = id_dataframe.values  # 加速度序列

        id_data = id_data[(id_data[:, 11] !=0)]
        id_data_temp = id_data[(id_data[:, 2] >= 347) & (id_data[:, 2] <= 390)]
        low_acc_data = id_data_temp[id_data_temp[:, 11] <= -11, :]
        high_acc_data = id_data_temp[id_data_temp[:, 11] >= 11, :]
        acc_data = np.row_stack((low_acc_data,high_acc_data))

        ## 判断轨段是否连续时间
        def split_segments(data):
            if data.shape[0] == 0:
                return []
            segments = []
            start_idx = 0
            for i in range(1, data.shape[0]):
                if data[i, 0] != data[i - 1, 0] + 1:  # 判断是否时间不连续
                    segments.append(data[start_idx:i])  # 保存当前连续段
                    start_idx = i  # 更新起始点
            segments.append(data[start_idx:])  # 保存最后一个连续段
            return segments

        segments = split_segments(acc_data)

        min_bounds, max_bounds = [], []
        for segment in segments:
            if segment.shape[0] >= 6:
                min_bounds.append(int(segment[:, 0].min()))
                max_bounds.append(int(segment[:, 0].max()))
        if min_bounds and max_bounds:
            min_bound = min(min_bounds)
            max_bound = max(max_bounds)
            selected_id_data = id_data[(id_data[:, 0] >= min_bound) & (id_data[:, 0] <= max_bound)]
            # if np.min(selected_id_data[:,2])<=367 and np.max(selected_id_data[:,2])>=367:  ##366.8
            if selected_id_data.shape[0] >= 15:
                selected_data = np.vstack((selected_data, selected_id_data))  # 合并到最终结果

    ####1.提取数据-方式2
    # for id, id_dataframe in grouped_dataframe:
    #     id_dataframe = id_dataframe.sort_values('frameNum')  # 按帧号排序
    #     id_data = id_dataframe.values  # 加速度序列
    #
    #     id_data_temp = id_data[(id_data[:, 2] > 350) & (id_data[:, 2] < 390)]
    #     low_acc_data = id_data_temp[id_data_temp[:, 5] <= -6, :]
    #     high_acc_data = id_data_temp[id_data_temp[:, 5] >= 5, :]
    #
    #     ## 判断轨段是否连续时间
    #     def split_segments(data):
    #         if data.shape[0] == 0:
    #             return []
    #         segments = []
    #         start_idx = 0
    #         for i in range(1, data.shape[0]):
    #             if data[i, 0] != data[i - 1, 0] + 1:  # 判断是否时间不连续
    #                 segments.append(data[start_idx:i])  # 保存当前连续段
    #                 start_idx = i  # 更新起始点
    #         segments.append(data[start_idx:])  # 保存最后一个连续段
    #         return segments
    #
    #     # 分段处理低加速度和高加速度轨迹
    #     low_segments = split_segments(low_acc_data)
    #     high_segments = split_segments(high_acc_data)
    #
    #     min_bounds, max_bounds = [], []
    #     for segment in low_segments + high_segments:
    #         if segment.shape[0] >= 5:
    #             min_bounds.append(int(segment[:, 0].min()))
    #             max_bounds.append(int(segment[:, 0].max()))
    #     if min_bounds and max_bounds:
    #         min_bound = min(min_bounds)
    #         max_bound = max(max_bounds)
    #         selected_id_data = id_data[(id_data[:, 0] >= min_bound) & (id_data[:, 0] <= max_bound)]
    #         selected_data = np.vstack((selected_data, selected_id_data))  # 合并到最终结果
    #########################################################################################
    #### 2.统计数据-当前车辆的数据
    data=dataframe.values
    for lane in np.unique(selected_data[:, 6]):
        lane_data = data[data[:, 6] == lane]
        lane_count = len(np.unique(lane_data[:, 1]))

        lane_selected_data=selected_data[selected_data[:, 6] == lane]
        selected_count = len(np.unique(lane_selected_data[:, 1]))
        proportion = selected_count / lane_count
        print('----')
        print(f'车道 {int(lane)} 车辆数量: {selected_count}')
        print(f'车道 {int(lane)} 轨迹占比: {proportion:.2%}')

        acc_values = lane_selected_data[:, 11]
        acc_75th_percentile = np.percentile(acc_values[acc_values > 0], 75) if np.any(acc_values > 0) else None
        dec_75th_percentile = np.percentile(acc_values[acc_values < 0], 25) if np.any(acc_values < 0) else None
        print(f'车道 {int(lane)} 75分位相对加速度: {acc_75th_percentile:.2f}')
        print(f'车道 {int(lane)} 75分位相对减速度: {dec_75th_percentile:.2f}')
        # 标准差和 RMSE
        mean_acc = np.mean(acc_values)
        acc_std = np.std(acc_values)
        mae = np.mean(np.abs(acc_values - mean_acc))
        print(f'车道 {int(lane)} 加速度均值: {mean_acc:.2f}')
        print(f'车道 {int(lane)} 加速度标准差: {acc_std:.2f}')
        print(f'车道 {int(lane)} 平均绝对误差 (MAE): {mae:.2f}')

    ## 所有车道统计（每次粘贴修改下）

    ###################################################################
    ## 统计数据-车辆对的数据
    data=dataframe.values
    selected_pairs_list = []
    for current_id in np.unique(selected_data[:, 1]):
        current_data = selected_data[selected_data[:, 1] == current_id]
        current_min_bound=current_data[:, 0].min()
        current_max_bound=current_data[:, 0].max()
        for preceding_id in np.unique(current_data[:, 8]):
            preceding_data = data[data[:, 1] == preceding_id]
            if preceding_data.shape[0] == 0:
                continue
            preceding_data_temp=preceding_data[(preceding_data[:, 2] >= 347) & (preceding_data[:, 2] <= 390)]
            if preceding_data_temp.shape[0] == 0:
                    continue
            t_temp =np.min(preceding_data_temp[:,0])
            # if preceding_data[0,6]<=3:
            #     preceding_data_temp=preceding_data[preceding_data[:,2]<=367]
            #     if preceding_data_temp.shape[0] == 0:
            #         continue
            #     t_temp =np.min(preceding_data_temp[:,0])
            # if preceding_data[0,6]>3:
            #     preceding_data_temp=preceding_data[preceding_data[:,2]<367]
            #     if preceding_data_temp.shape[0] == 0:
            #         continue
            #     t_temp =np.min(preceding_data_temp[:,0])
            min_bound_ = t_temp
            max_bound_ = current_max_bound
            data_temp = data[(data[:, 0] > min_bound_) & (data[:, 0] < max_bound_)]
            data_temp = data_temp[(data_temp[:, 1] == current_id) & (data_temp[:, 8] == preceding_id)]
            selected_pairs_list.append(data_temp)
    # 统计车辆对的数量，比例，时长，
    for lane in np.unique(data[:, 6]):  # 遍历所有车道
        lane_data = data[data[:, 6] == lane]  # 当前车道数据
        if lane_data.shape[0] == 0:
            continue
        lane_data=lane_data[lane_data[:,8]!=0,:]
        lane_pairs = len(np.unique(lane_data[:, 1]))  # 每个车道车辆对的对数

        lane_selected_pairs = 0
        pairs_times,pairs_snp = [], []
        for _,temp_pair in enumerate(selected_pairs_list):
            if temp_pair.shape[0]==0:
                continue
            if temp_pair[0,6]==lane: # 当前车辆对属于该车道
                lane_selected_pairs += 1  # 累计车辆对数量
                temp_pair = temp_pair[np.argsort(temp_pair[:, 0])]
                time_ = temp_pair[-1, 0] - temp_pair[0, 0] # 时长
                pairs_times.append(time_)
                pairs_snp.extend(temp_pair[:, 12].tolist())
        proportion = lane_selected_pairs / lane_pairs
        avg_time = np.mean(pairs_times)/30
        avg_snp = np.mean(pairs_snp)
        print(f"车道 {int(lane)} 统计结果：")
        print(f"  车辆对数量: {lane_selected_pairs:}")
        print(f"  车辆对数量占比: {proportion:.2%}")
        print(f"  平均通过时长: {avg_time:.2f}")
        print(f"  平均车间距: {avg_snp:.2f}")

    ## 所有车道统计（每次粘贴修改下）
    lane_data = data # 当前车道数据
    lane_data=lane_data[lane_data[:,8]!=0,:]
    lane_pairs = len(np.unique(lane_data[:, 1]))  # 每个车道车辆对的对数
    lane_selected_pairs = 0
    pairs_times,pairs_snp = [], []
    for _,temp_pair in enumerate(selected_pairs_list):
        if temp_pair.shape[0]==0:
            continue
        lane_selected_pairs += 1  # 累计车辆对数量
        temp_pair = temp_pair[np.argsort(temp_pair[:, 0])]
        time_ = temp_pair[-1, 0] - temp_pair[0, 0] # 时长
        pairs_times.append(time_)
        pairs_snp.extend(temp_pair[:, 12].tolist())
    proportion = lane_selected_pairs / lane_pairs
    avg_time = np.mean(pairs_times)/30
    avg_snp = np.mean(pairs_snp)
    print(f"车道999统计结果：")
    print(f"  999车辆对数量: {lane_selected_pairs:}")
    print(f"  999车辆对数量占比: {proportion:.2%}")
    print(f"  999平均通过时长: {avg_time:.2f}")
    print(f"  999平均车间距: {avg_snp:.2f}")
    ###############################################################
    ####3.绘图
    # selected_data=selected_data[selected_data[:,6]==5]
    plt.figure(figsize=(12, 8))
    unique_car_ids = np.unique(selected_data[:, 1])
    for car_id in unique_car_ids:
        # selected_data = selected_data[selected_data[:, 6] == 3]
        car_data = selected_data[selected_data[:, 1] == car_id]
        car_data = car_data[np.argsort(car_data[:, 0])]
        # time = car_data[:, 2]
        # middle_index = len(car_data) // 2
        # distance = car_data[:, 10]-car_data[0, 10]
        # plt.plot(time, distance)
        frame_nums = car_data[:, 2]
        accelerations = car_data[:, 5]
        plt.plot(frame_nums, accelerations, label=f'Car ID: {int(car_id)}', linewidth=1)

    # for i, data_temp in enumerate(selected_pairs_list):
    #     if data_temp.shape[0] == 0:
    #         continue
    #     car_data = data_temp
    #     frame_nums = car_data[:, 2]  # 帧号
    #     accelerations = car_data[:, 4]  # 加速度
    #     plt.plot(frame_nums, accelerations, linewidth=1)
    plt.show()