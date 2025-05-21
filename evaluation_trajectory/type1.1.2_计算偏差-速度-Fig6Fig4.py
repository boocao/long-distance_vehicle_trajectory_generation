import os
import re
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
import seaborn as sns
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# 导入字体属性相关的包或类
from matplotlib.font_manager import FontProperties

if __name__=='__main__':
    ####数据:time,id,x,y,w,h,vx,vy,v,ax,ay,a,Speed2D-(m/s),VelForward-(m/s),VelLateral-(m/s) # 6-8,9-11,12-
                                            #Accel-,AccelForward-(m/s_), AccelLateral-(m/s_) # 12-
    data_df1 = pd.read_csv('./csvs/rts_exp0606_truck0002_velocity.csv')
    data_df2 = pd.read_csv('./csvs/rts_exp0607_truck0003_velocity.csv')

    data1 = data_df1.values
    data2 = data_df2.values

    # 按条件过滤数据
    data1_1 = data1[data1[:, 1] == 3, :]
    data1_2 = data1[data1[:, 1] == 5, :]
    data2_1 = data2[data2[:, 1] == 3, :]
    data2_2 = data2[data2[:, 1] == 5, :]
    data_all = np.row_stack((data1_1, data1_2, data2_1, data2_2))

    # 过滤掉无效数据（12列为-1）
    data1_1 = data1_1[data1_1[:, 12] != -1, :]
    data2_2 = data2_2[data2_2[:, 12] != -1, :]
    data_all = data_all[data_all[:, 12] != -1, :]
    bias_v_1_1 = data1_1[:, 8]-data1_1[:, 12]
    bias_v_1_2 = data1_2[:, 8]-data1_2[:, 12]
    bias_v_2_1 = data2_1[:, 8]-data2_1[:, 12]
    bias_v_2_2 = data2_2[:, 8]-data2_2[:, 12]
    bias_v_all = data_all[:, 8]-data_all[:, 12]

    bias_stat = bias_v_all
    y_i = data_all[:, 12]

    size = bias_stat.shape[0]
    max_value = bias_stat.max()
    min_value = bias_stat.min()
    mean_bias = bias_stat.mean()
    std_bias = bias_stat.std()
    rmse = np.sqrt(np.mean((bias_stat) ** 2))
    rmspe = np.sqrt(np.mean((bias_stat / y_i) ** 2)) * 100

    mae = np.mean(np.abs(bias_stat))  # 平均绝对误差
    mape = np.mean(np.abs(bias_stat / y_i)) * 100 # 平均绝对百分比误差
    wape = (np.sum(np.abs(bias_stat)) / np.sum(np.abs(y_i))) * 100  # 加权绝对百分比误差
    cv = (std_bias / mean_bias) * 100

    # 打印结果，保留两位小数
    print(f"数据量 (size): {size}")
    print(f"最大值 (max): {max_value:.2f}")
    print(f"最小值 (min): {min_value:.2f}")
    print(f"均值 (mean): {mean_bias:.2f}")
    print(f"标准差 (std): {std_bias:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"RMSPE: {rmspe:.2f} %")
    # print(f"MAE: {mae:.2f}")
    # print(f"MAPE: {mape:.2f} %")
    # print(f"WAPE: {wape:.2f} %")
    # print(f"变异系数 (CV): {cv:.2f} %")

    ###偏差随位置，随长度
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(14, 3.))
    colors = ['RoyalBlue', 'DodgerBlue', 'green', 'LimeGreen']
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

    ax2.plot(data1_2[:, 2], bias_v_1_2, c=colors[1], label='group 2')
    ax2.plot(data1_1[:, 2], bias_v_1_1, c=colors[0], label='group 1')
    ax2.plot(data2_2[:, 2], bias_v_2_2, c=colors[3], label='group 4')
    ax2.plot(data2_1[:, 2], bias_v_2_1, c=colors[2], label='group 3')

    # ax2.scatter(data1_1[:, 8], bias_v_1_1, c=colors[0], label='group 1')
    # ax2.scatter(data1_2[:, 8], bias_v_1_2, c=colors[1], label='group 2')
    # ax2.scatter(data2_1[:, 8], bias_v_2_1, c=colors[2], label='group 3')
    # ax2.scatter(data2_2[:, 8], bias_v_2_2, c=colors[3], label='group 4')

    def sliding_window_filter(data, window_size):
        """使用滑动窗口进行均值滤波"""
        filtered_data = np.copy(data)
        half_window = window_size // 2
        for i in range(len(data)):
            start = max(0, i - half_window)
            end = min(len(data), i + half_window + 1)
            filtered_data[i] = np.mean(data[start:end])
        return filtered_data

    # 设置窗口大小
    window_size = 101
    data_all = data_all[data_all[:, 2].argsort()]
    bias_v_all = data_all[:, 8] - data_all[:, 12]
    bias_v_all_filtered = sliding_window_filter(bias_v_all, window_size)
    ax2.plot(data_all[:, 2], bias_v_all_filtered, linewidth=1, c='red', label='all groups')

    # 设置第一个子图的标签和样式
    # ax2.legend(loc='upper center', prop={'family': 'Times New Roman',"size": 14}, frameon=False, ncol=2)
    ax2.set_xlabel("Distance (m)",fontname='Times New Roman', fontsize=20)
    ax2.set_ylabel("Velocity bias (m/s)",fontname='Times New Roman', fontsize=20)
    ax2.set_xticks(np.arange(0, 350, 50))
    ax2.set_yticks(np.arange(-1.5, 1.5, 0.5))
    ax2.tick_params(axis='both', labelsize=20)
    ax2.grid(True)
    ##### 绘制概率密度图（第二个子图）
    bins = np.arange(-2.125, 1.5, 0.25)
    counts1_1, bin_edges = np.histogram(bias_v_1_1, bins=bins, density=False)
    counts1_2, bin_edges = np.histogram(bias_v_1_2, bins=bins, density=False)
    counts2_1, bin_edges = np.histogram(bias_v_2_1, bins=bins, density=False)
    counts2_2, bin_edges = np.histogram(bias_v_2_2, bins=bins, density=False)
    counts_all, bin_edges = np.histogram(bias_v_all, bins=bins, density=False)

    # # 归一化概率密度
    counts1_1 = counts1_1.astype(np.float64)
    counts1_2 = counts1_2.astype(np.float64)
    counts2_1 = counts2_1.astype(np.float64)
    counts2_2 = counts2_2.astype(np.float64)
    counts_all = counts_all.astype(np.float64)
    counts1_1 /= counts1_1.sum()
    counts1_2 /= counts1_2.sum()
    counts2_1 /= counts2_1.sum()
    counts2_2 /= counts2_2.sum()
    counts_all /= counts_all.sum()

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax1.plot(bin_centers, counts1_1, linestyle='-', marker='', color=colors[0], label='Group 1')
    ax1.plot(bin_centers, counts1_2, linestyle='-', marker='', color=colors[1], label='Group 2')
    ax1.plot(bin_centers, counts2_1, linestyle='-', marker='', color=colors[2], label='Group 3')
    ax1.plot(bin_centers, counts2_2, linestyle='-', marker='', color=colors[3], label='Group 4')

    # ax1.legend(loc='upper right',prop={'family': 'Times New Roman',"size": 18}, frameon=False)
    ax1.set_xlabel("Velocity bias (m/s)",fontname='Times New Roman', fontsize=20)
    ax1.set_ylabel("Probability density",fontname='Times New Roman', fontsize=20)
    ax1.set_xticks(np.arange(-2.5, 1.5, 0.5))
    ax1.set_yticks(np.arange(0, 0.5, 0.1))
    ax1.tick_params(axis='both', labelsize=20)
    ax1.grid(True)

    # plt.tight_layout()
    plt.savefig('./imgs/figure2.png', dpi=900)
   #################################################################################
    data_df1 = pd.read_csv('./csvs/rts_exp0606_truck0002_velocity.csv')
    data_df2 = pd.read_csv('./csvs/rts_exp0607_truck0003_velocity.csv')

    data1 = data_df1.values
    data2 = data_df2.values
    data1_1 = data1[data1[:, 1] == 3, :]
    data1_2 = data1[data1[:, 1] == 5, :]
    data2_1 = data2[data2[:, 1] == 3, :]
    data2_2 = data2[data2[:, 1] == 5, :]
    data1_1 = data1_1[data1_1[:, 12] != -1, :]
    data2_2 = data2_2[data2_2[:, 12] != -1, :]

    data1_1[:, 0]=data1_1[:, 0]-np.min(data1_1[:, 0])
    data1_2[:, 0]=data1_2[:, 0]-np.min(data1_2[:, 0])
    data2_1[:, 0]=data2_1[:, 0]-np.min(data2_1[:, 0])
    data2_2[:, 0]=data2_2[:, 0]-np.min(data2_2[:, 0])


    # 创建画布和子图
    fig, axs = plt.subplots(2, 2, figsize=(7, 6))  # 增大画布尺寸
    colors = ['RoyalBlue', 'DodgerBlue', 'green', 'LimeGreen']  # 自定义颜色
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    # 绘制图形
    axs[0, 0].plot(data1_1[:, 0], data1_1[:, 8], color=colors[0], label='group 1')
    axs[0, 0].plot(data1_1[:, 0], data1_1[:, 12], color='orange', label='truck')
    # axs[0, 0].set_title("Group 1", fontsize=20)
    # axs[0, 0].set_xlabel("Time (s)", fontsize=20)
    axs[0, 0].set_ylabel("Velocity (m/s)", fontsize=20)
    axs[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(5))
    axs[0, 0].yaxis.set_major_locator(ticker.MultipleLocator(1))
    axs[0, 0].tick_params(axis='both', which='major', labelsize=20)
    # axs[0, 0].legend(fontsize=20)

    axs[0, 1].plot(data1_2[:, 0], data1_2[:, 8], color=colors[1], label='group 2')
    axs[0, 1].plot(data1_2[:, 0], data1_2[:, 12], color='orange', label='truck')
    # axs[0, 1].set_title("Group 2", fontsize=20)
    # axs[0, 1].set_xlabel("Time (s)", fontsize=20)
    axs[0, 1].set_ylabel("Velocity (m/s)", fontsize=20)
    axs[0, 1].xaxis.set_major_locator(ticker.MultipleLocator(5))
    axs[0, 1].yaxis.set_major_locator(ticker.MultipleLocator(1))
    axs[0, 1].tick_params(axis='both', which='major', labelsize=20)
    # axs[0, 1].legend(fontsize=20)

    axs[1, 0].plot(data2_1[:, 0], data2_1[:, 8], color=colors[2], label='group 3')
    axs[1, 0].plot(data2_1[:, 0], data2_1[:, 12], color='orange', label='truck')
    # axs[1, 0].set_title("Group 3", fontsize=20)
    axs[1, 0].set_xlabel("Time (s)", fontsize=20)
    axs[1, 0].set_ylabel("Velocity (m/s)", fontsize=20)
    axs[1, 0].xaxis.set_major_locator(ticker.MultipleLocator(5))
    axs[1, 0].yaxis.set_major_locator(ticker.MultipleLocator(1))
    axs[1, 0].tick_params(axis='both', which='major', labelsize=20)
    # axs[1, 0].legend(fontsize=20)

    axs[1, 1].plot(data2_2[:, 0], data2_2[:, 8], color=colors[3], label='group 4')
    axs[1, 1].plot(data2_2[:, 0], data2_2[:, 12], color='orange', label='truck')
    # axs[1, 1].set_title("Group 4", fontsize=20)
    axs[1, 1].set_xlabel("Time (s)", fontsize=20)
    axs[1, 1].set_ylabel("Velocity (m/s)", fontsize=20)
    axs[1, 1].xaxis.set_major_locator(ticker.MultipleLocator(5))
    axs[1, 1].yaxis.set_major_locator(ticker.MultipleLocator(1))
    axs[1, 1].tick_params(axis='both', which='major', labelsize=20)
    # axs[1, 1].legend(fontsize=20)

    # 设置标签
    for ax in axs.flat:
        ax.tick_params(axis='both', which='major', labelsize=20)
    # 提取所有子图的图例句柄和标签
    # handles = []
    # labels = []
    # for ax in axs.flat:
    #     h, l = ax.get_legend_handles_labels()
    #     handles.extend(h)
    #     labels.extend(l)
    # fig.legend(handles, labels, loc='upper center',prop={'family': 'Times New Roman', 'size': 20}, ncol=5, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig('./imgs/figure.png', dpi=900)


