
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from PIL import Image

def draw_xy(data,col1,col2):
    colours = np.random.rand(32, 3)
    plt.figure(figsize=(10,3))
    for i in range(int(data[:, 1].max())):
        j = i + 1
        id_data = data[data[:, 1] == j, :]
        x,y=id_data[:,col1],id_data[:,col2]
        # plt.plot(x, y, linewidth= ,color=colours[j % 32, :])
        # plt.plot(x, y,linewidth=4,color='#1E90FF')
        # plt.plot(x, y, linewidth=4,color='#708090')
        plt.plot(x, y, linestyle='-',linewidth=2,color='black')

    plt.xticks(fontproperties='Times New Roman',size=20)
    plt.yticks([])
    # plt.xlabel("Frame",fontproperties='Times New Roman',size=20)
    # plt.ylabel("Distance (m)",fontproperties='Times New Roman',size=20)
    # plt.show()
    plt.savefig('test.png', format='png', transparent=True, dpi=900)


if __name__=='__main__':
    txt_path1 ='stitching-exp0502exp4id=4.txt'
    txt_path2 ='stitching-exp0502exp5id=51.txt'
    txt_path3 ='stitching-exp0502id=15.txt'
    data1 = np.loadtxt(txt_path1, delimiter=',',dtype=bytes).astype(str)
    data2 = np.loadtxt(txt_path2, delimiter=',',dtype=bytes).astype(str)
    data3 = np.loadtxt(txt_path3, delimiter=',',dtype=bytes).astype(str)
    data1 = data1.astype(np.float64)
    data2 = data2.astype(np.float64)
    data3 = data3.astype(np.float64)

    data1 = data1[data1[:, 2] >= 2416, :]
    data1 = data1[data1[:, 2] <= 3416+50, :]

    data2[:, 2] =data2[:, 2]+3416-200
    data2 = data2[data2[:, 2] >= 3416-50, :]
    data2 = data2[data2[:, 2] <= 4416, :]

    data3 = data3[data3[:, 2] >= 2416, :]
    data3 = data3[data3[:, 2] <= 4416, :]

    plt.figure(figsize=(10,3))
    x1,y1=data1[:,2],data1[:,3]-5
    x2,y2=data2[:,2],data2[:,3]+5
    x3,y3=data3[:,2],data3[:,3]
    # plt.plot(x1, y1,linewidth=4,color='#1E90FF')
    # plt.plot(x2, y2, linewidth=4,color='#708090')
    # plt.plot(x3, y3,linewidth=4,color='#1E90FF')
    plt.plot(x1, y1, linestyle='-',linewidth=2,color='gray')
    plt.plot(x2, y2, linestyle='-',linewidth=2,color='gray')
    plt.plot(x3, y3, linestyle='-',linewidth=2,color='black')

    plt.xticks(fontproperties='Times New Roman',size=20)
    plt.yticks([])
    # plt.xlabel("Frame",fontproperties='Times New Roman',size=20)
    # plt.ylabel("Distance (m)",fontproperties='Times New Roman',size=20)
    # plt.show()
    plt.savefig('test.png', format='png', transparent=True, dpi=900)

    

