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
# 导入字体属性相关的包或类
from matplotlib.font_manager import FontProperties

if __name__=='__main__':
    data_df1 = pd.read_csv('./csvs/rts_exp0606_truck0002_velocity.csv')
    data_df2 = pd.read_csv('./csvs/rts_exp0607_truck0003_velocity.csv')

    data1 = data_df1.values
    data2 = data_df2.values
    data1_1 = data1[data1[:, 1] == 3, :]
    data1_2 = data1[data1[:, 1] == 5, :]
    data2_1 = data2[data2[:, 1] == 3, :]
    data2_2 = data2[data2[:, 1] == 5, :]
    data_all = np.row_stack((data1_1, data1_2, data2_1, data2_2))

    # 车身数据统计
    data1_1 = data1_1[data1_1[:, 12] != -1, :]
    data2_2 = data2_2[data2_2[:, 12] != -1, :]
    data_all = data_all[data_all[:, 12] != -1, :]
    bias_l_1_1 = data1_1[:, 4] - 12
    bias_l_1_2 = data1_2[:, 4] - 12
    bias_l_2_1 = data2_1[:, 4] - 12
    bias_l_2_2 = data2_2[:, 4] - 12
    bias_l_all = data_all[:, 4]-12

    bias_stat = bias_l_all
    size = bias_stat.shape[0]
    max_value = bias_stat.max()
    min_value = bias_stat.min()
    mean_bias = bias_stat.mean()
    std_bias = bias_stat.std()
    rmse = np.sqrt(np.mean((bias_stat) ** 2))
    rmspe = np.sqrt(np.mean((bias_stat / 12) ** 2)) * 100

    mae = np.mean(np.abs(bias_stat))  # 平均绝对误差
    mape = np.mean(np.abs(bias_stat / 12)) * 100 # 平均绝对百分比误差
    wape = (np.sum(np.abs(bias_stat)) /(np.abs(12)*size)) * 100  # 加权绝对百分比误差
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

    # 绘图：偏差随位置，随长度
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(14, 3))  # 创建1行2列的子图
    colors = ['RoyalBlue', 'DodgerBlue', 'green', 'LimeGreen']
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

    # 绘制偏差随位置变化的曲线
    ax2.plot(data1_2[:, 2], bias_l_1_2, linewidth=0.8, c=colors[1], label='group 2')
    ax2.plot(data1_1[:, 2], bias_l_1_1, linewidth=0.8, c=colors[0], label='group 1')
    ax2.plot(data2_2[:, 2], bias_l_2_2, linewidth=0.8, c=colors[3], label='group 4')
    ax2.plot(data2_1[:, 2], bias_l_2_1, linewidth=0.8, c=colors[2], label='group 3')

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
    bias_l_all = data_all[:, 4] - 12
    bias_l_all_filtered = sliding_window_filter(bias_l_all, window_size)

    # 绘制滤波后的曲线
    ax2.plot(data_all[:, 2], bias_l_all_filtered, linewidth=1, c='red', label='all groups')

    # 设置图例、标签和刻度
    # ax2.legend(loc='upper center', prop={'family': 'Times New Roman',"size": 14}, frameon=False, ncol=2)
    ax2.set_xlabel("Distance (m)",fontname='Times New Roman', fontsize=20)
    ax2.set_ylabel("Length bias (m)",fontname='Times New Roman', fontsize=20)
    ax2.set_xticks(np.arange(0, 350, 50))
    ax2.set_yticks(np.arange(0, 4, 1))
    ax2.tick_params(axis='both', labelsize=20)
    ax2.grid(True)

    # 第二幅图：概率密度
    bins = np.arange(-1.25, 3.5+0.5, 0.5)
    counts1_1, bin_edges = np.histogram(bias_l_1_1, bins=bins, density=False)
    counts1_2, bin_edges = np.histogram(bias_l_1_2, bins=bins, density=False)
    counts2_1, bin_edges = np.histogram(bias_l_2_1, bins=bins, density=False)
    counts2_2, bin_edges = np.histogram(bias_l_2_2, bins=bins, density=False)
    counts_all, bin_edges = np.histogram(bias_l_all, bins=bins, density=False)

    # 归一化
    counts1_1 = counts1_1 / counts1_1.sum()
    counts1_2 = counts1_2 / counts1_2.sum()
    counts2_1 = counts2_1 / counts2_1.sum()
    counts2_2 = counts2_2 / counts2_2.sum()
    counts_all = counts_all / counts_all.sum()

    # 计算每个bin的中心
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax1.plot(bin_centers, counts1_1, linestyle='-', marker='', color=colors[0], label='group 1')
    ax1.plot(bin_centers, counts1_2, linestyle='-', marker='', color=colors[1], label='group 2')
    ax1.plot(bin_centers, counts2_1, linestyle='-', marker='', color=colors[2], label='group 3')
    ax1.plot(bin_centers, counts2_2, linestyle='-', marker='', color=colors[3], label='group 4')
    ax1.plot(bin_centers, counts_all, linestyle='-', marker='', color='red', label='all groups')

    # 设置图例、标签和刻度
    # ax1.legend(loc='upper right',prop={'family': 'Times New Roman',"size": 18}, frameon=False)
    ax1.set_xlabel("Length bias (m)",fontname='Times New Roman', fontsize=20)
    ax1.set_ylabel("Probability density",fontname='Times New Roman', fontsize=20)
    ax1.set_xticks(np.arange(-1, 4, 0.5))
    ax1.set_yticks(np.arange(0, 0.6, 0.1))
    ax1.tick_params(axis='both', labelsize=20)
    ax1.grid(True)

    # handles, labels = ax1.get_legend_handles_labels()
    # fig.legend(handles[:5], labels[:5],loc='upper center', prop={'family': 'Times New Roman', 'size': 20}, ncol=5, frameon=False)
    # plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig('./imgs/figure1.png', dpi=900)