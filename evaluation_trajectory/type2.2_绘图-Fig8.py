import matplotlib
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

if __name__ == "__main__":
    ## ['frameNum', 'carId', 'carCenterXm', 'carCenterYm', 'speed', 'acceleration',
    ##  'laneId', 'length', 'preceding_id', 'following_id',
    ## 'delta_v', 'delta_a', 'obs_snp'] #10,11,12
    # 第一组数据
    dataframe1 = pd.read_csv('./CitySIM/FreewayC01_for_snp.csv', header=0)
    data1 = dataframe1.values

    x_list1 = []
    v_list1 = []
    linewidth = 2
    maxx1 = int(max(data1[:, 2]))
    n1 = len(np.unique(data1[:, 6]))

    for l in np.unique(data1[:, 6]):
        lane_data1 = data1[data1[:, 6] == l, :]
        for a in np.arange(1, maxx1, linewidth):
            temp_data1 = lane_data1[(lane_data1[:, 2] >= a) & (lane_data1[:, 2] < (a + linewidth))]
            if temp_data1.shape[0] > 0:
                x_list1.append(a + linewidth - 1)
                v_list1.append(temp_data1[:, 5].mean())
            else:
                x_list1.append(a + linewidth - 1)
                v_list1.append(np.nan)

    x_array1 = np.array(x_list1).reshape((n1, -1))
    v_array1 = np.array(v_list1).reshape((n1, -1))

    # 第二组数据
    dataframe2 = pd.read_csv('./CitySIM/FreewayC01_snp_data.csv', header=0)
    data2 = dataframe2.values

    x_list2 = []
    v_list2 = []
    maxx2 = int(max(data2[:, 2]))
    n2 = len(np.unique(data2[:, 6]))

    for l in np.unique(data2[:, 6]):
        lane_data2 = data2[data2[:, 6] == l, :]
        for a in np.arange(1, maxx2, linewidth):
            temp_data2 = lane_data2[(lane_data2[:, 2] >= a) & (lane_data2[:, 2] < (a + linewidth))]
            if temp_data2.shape[0] > 0:
                x_list2.append(a + linewidth - 1)
                v_list2.append(temp_data2[:, 11].mean())
            else:
                x_list2.append(a + linewidth - 1)
                v_list2.append(np.nan)

    x_array2 = np.array(x_list2).reshape((n2, -1))
    v_array2 = np.array(v_list2).reshape((n2, -1))

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.7))  # 创建14x3.7的画布，1行2列
    cmap = plt.cm.get_cmap('viridis')  # 选择色彩映射
    norm = plt.Normalize(vmin=1, vmax=10)  # 设置数值范围
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

    # 第一张图
    for l in range(n1):
        if l == 0 or l == 7:
            continue
        x_data1 = x_array1[l, :]
        y_data1 = v_array1[l, :]
        axes[0].plot(x_data1, y_data1, linewidth=1.5, color=cmap(norm(8 - l)), label="lane " + str(l))
    axes[0].set_xlim([300, 445])
    axes[0].xaxis.set_major_locator(ticker.MultipleLocator(20))  # 设置x轴刻度间隔为10
    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(10))  # 设置y轴刻度间隔为5
    axes[0].set_xlabel("X (m)", fontproperties='Times New Roman', size=20)
    axes[0].set_ylabel("Acceleration (m/s²)", fontproperties='Times New Roman', size=20)
    axes[0].tick_params(axis='both', labelsize=20)  # 设置刻度字体大小
    # axes[0].grid(True, linewidth=0.5, alpha=0.5)

    # 添加灰色参考线
    axes[0].axhline(y=5, color='gray', linestyle='--', linewidth=1,alpha=0.9)
    axes[0].axhline(y=-6, color='gray', linestyle='--', linewidth=1,alpha=0.9)
    axes[0].axvline(x=345, color='gray', linestyle='--', linewidth=1,alpha=0.9)
    axes[0].axvline(x=390, color='gray', linestyle='--', linewidth=1,alpha=0.9)

    # 第二张图
    for l in range(n2):
        if l == 0 or l == 7:
            continue
        x_data2 = x_array2[l, :]
        y_data2 = v_array2[l, :]
        axes[1].plot(x_data2, y_data2, linewidth=1.5, color=cmap(norm(8 - l)), label="lane" + str(l))
    axes[1].set_xlim([300, 445])
    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(20))  # 设置x轴刻度间隔为10
    axes[1].yaxis.set_major_locator(ticker.MultipleLocator(10))  # 设置y轴刻度间隔为5
    axes[1].set_xlabel("X (m)", fontproperties='Times New Roman', size=20)
    axes[1].set_ylabel("Acceleration difference (m/s²)", fontproperties='Times New Roman', size=20)
    axes[1].tick_params(axis='both', labelsize=20)  # 设置刻度字体大小
    # axes[1].grid(True, linewidth=0.5, alpha=0.5)

    # 添加灰色参考线
    axes[1].axhline(y=11, color='gray', linestyle='--', linewidth=1,alpha=0.9)
    axes[1].axhline(y=-11, color='gray', linestyle='--', linewidth=1,alpha=0.9)
    axes[1].axvline(x=345, color='gray', linestyle='--', linewidth=1,alpha=0.9)
    axes[1].axvline(x=390, color='gray', linestyle='--', linewidth=1,alpha=0.9)

    # handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles[:6], labels[:6],loc='upper center', prop={'family': 'Times New Roman', 'size': 20},
    #            ncol=6, frameon=False, handlelength=1.8)
    # plt.tight_layout()  # 调整布局
    # plt.show()
    plt.savefig('./imgs/figure_combined.png', dpi=600)
