import matplotlib
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ## ['frameNum', 'carId', 'carCenterXm', 'carCenterYm', 'speed', 'acceleration',
    ##  'laneId','length', 'preceding_id', 'following_id','obs_snp','delta_v']
    dataframe=pd.read_csv('./CitySIM/FreewayC01_for_snp.csv',header=0)
    grouped_dataframe = dataframe.groupby('carId')
    ## 1.提取偏差的轨迹数据
    selected_data=np.empty((0, dataframe.shape[1]))
    for id, id_dataframe in grouped_dataframe:
        id_dataframe = id_dataframe.sort_values('frameNum')  # 按帧号排序
        id_data = id_dataframe.values  # 加速度序列

        id_data_temp = id_data[(id_data[:, 2] >= 347) & (id_data[:, 2] <= 390)]
        low_acc_data = id_data_temp[id_data_temp[:, 5] <= -6, :]
        high_acc_data = id_data_temp[id_data_temp[:, 5] >= 5, :]
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

        min_bounds,max_bounds= [], []
        for segment in segments:
            if segment.shape[0] >= 6:
                min_bounds.append(int(segment[:, 0].min()))
                max_bounds.append(int(segment[:, 0].max()))
        if min_bounds and max_bounds:
            min_bound = min(min_bounds)
            max_bound = max(max_bounds)
            selected_id_data=id_data[(id_data[:, 0] >= min_bound) & (id_data[:, 0] <= max_bound)]
            # if np.min(selected_id_data[:,2])<=367 and np.max(selected_id_data[:,2])>=367:  ##366.8
            if selected_id_data.shape[0] >= 15:
                selected_data = np.vstack((selected_data, selected_id_data))  # 合并到最终结果
    ######################################################################################
    ##2.统计数据
    data=dataframe.values
    for lane in np.unique(selected_data[:, 6]):
        lane_data = data[data[:, 6] == lane]
        lane_count = len(np.unique(lane_data[:, 1]))

        lane_selected_data=selected_data[selected_data[:, 6] == lane]
        selected_count = len(np.unique(lane_selected_data[:, 1]))
        proportion = selected_count / lane_count
        print('----')
        print(f'车道 {int(lane)} 轨迹数量: {selected_count}')
        print(f'车道 {int(lane)} 轨迹占比: {proportion:.2%}')

        durations = []
        distances = []
        for car_id in np.unique(lane_selected_data[:, 1]):
            car_data = lane_selected_data[lane_selected_data[:, 1] == car_id]
            durations.append(car_data[:, 0].max() - car_data[:, 0].min() + 1)
            distances.append(car_data[:, 2].max() - car_data[:, 2].min())
        avg_duration = np.mean(durations)/30 if durations else 0
        avg_distance = np.mean(distances) if distances else 0
        print(f'车道 {int(lane)} 平均时长: {avg_duration:.2f}')
        print(f'车道 {int(lane)} 平均行驶距离: {avg_distance:.2f}')
        # 75分位加速度和减速度
        acc_values = lane_selected_data[:, 5]
        acc_75th_percentile = np.percentile(acc_values[acc_values > 0], 75) if np.any(acc_values > 0) else None
        dec_75th_percentile = np.percentile(acc_values[acc_values < 0], 25) if np.any(acc_values < 0) else None
        print(f'车道 {int(lane)} 75分位加速度: {acc_75th_percentile:.2f}')
        print(f'车道 {int(lane)} 75分位减速度: {dec_75th_percentile:.2f}')
        # 标准差和 RMSE
        mean_acc = np.mean(acc_values)
        acc_std = np.std(acc_values)
        mae = np.mean(np.abs(acc_values - mean_acc))
        wape = (np.sum(np.abs(acc_values - mean_acc)) / np.sum(np.abs(acc_values))) * 100
        cv = (acc_std / mean_acc) * 100
        print(f'车道 {int(lane)} 加速度均值: {mean_acc:.2f}')
        print(f'车道 {int(lane)} 加速度标准差: {acc_std:.2f}')
        print(f'车道 {int(lane)} 平均绝对误差 (MAE): {mae:.2f}')
        # print(f'车道 {int(lane)} 加权绝对百分比误差 (WAPE): {"无效" if np.isnan(wape) else f"{wape:.2f} %"}')
        # print(f'车道 {int(lane)} 变异系数 (CV): {cv:.2f} %')

    ## 所有车道统计（每次粘贴修改下）

    ########################################################
    ########################################################
    ##3.画图
    plt.figure(figsize=(12, 8))
    # selected_data=selected_data[selected_data[:,6]==5]
    unique_car_ids = np.unique(selected_data[:, 1])
    for car_id in unique_car_ids:
        car_data = selected_data[selected_data[:, 1] == car_id]
        frame_nums = car_data[:, 2]  # 帧号
        accelerations = car_data[:, 5]  # 加速度
        plt.plot(frame_nums, accelerations, label=f'Car ID: {int(car_id)}', linewidth=1)
    plt.show()
    #######################################################
    # fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    # selected_data[:, 2] = (selected_data[:, 2] // 1) * 1
    # unique_lanes = np.unique(selected_data[:, 6])
    # for lane in unique_lanes:
    #     lane_data = selected_data[selected_data[:, 6] == lane]
    #     x_list, y_list = [], []
    #     for x in np.sort(lane_data[:, 2]):
    #         per_x_data = lane_data[lane_data[:, 2] == x, :]
    #         if per_x_data.shape[0] == 0:
    #             continue
    #         y = per_x_data[:, 4].mean()
    #         x_list.append(x)
    #         y_list.append(y)
    #     axes[0].plot(x_list, y_list, label=f'Lane {int(lane)}')
    # axes[0].set_title('Speed vs Position by Lane')
    # axes[0].set_xlabel('Position (X)')
    # axes[0].set_ylabel('Speed')
    # axes[0].legend()
    # axes[0].grid()
    #
    # # 加速度沿位置曲线
    # for lane in unique_lanes:
    #     lane_data = selected_data[selected_data[:, 6] == lane]
    #     x_list, y_list = [], []
    #     for x in np.sort(np.unique(lane_data[:, 2])):
    #         per_x_data = lane_data[lane_data[:, 2] == x, :]
    #         if per_x_data.shape[0] == 0:
    #             continue
    #         y = per_x_data[:, 5].mean()
    #         x_list.append(x)
    #         y_list.append(y)
    #     axes[1].plot(x_list, y_list, label=f'Lane {int(lane)}')
    # axes[1].set_title('Acceleration vs Position by Lane')
    # axes[1].set_xlabel('Position (X)')
    # axes[1].set_ylabel('Acceleration')
    # axes[1].legend()
    # axes[1].grid()
    #
    # plt.tight_layout()
    # plt.show()



