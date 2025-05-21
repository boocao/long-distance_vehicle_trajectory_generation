
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from PIL import Image


def draw_points(data,col1,col2,img_path=None):
    if img_path is not None:
        img = Image.open(img_path)
        plt.imshow(img)
    plt.figure(figsize=(9, 6))
    # plt.axis([0, 3840, 0, 2160])
    plt.scatter(data[:,col1],data[:,col2],c='black',s=1,linewidths=1,label='trajectory point')
    plt.title("Trajectory point")
    plt.ylabel("Y(m))")
    plt.xlabel("X(m))")
    plt.legend()
    plt.show()

def draw_xy(data,col1,col2):
    colours = np.random.rand(32, 3)
    plt.figure(figsize=(9, 6))
    # plt.figure(figsize=(12, 6))
    for i in range(int(data[:, 1].max())):
        j = i + 1
        id_data = data[data[:, 1] == j, :]
        x,y=id_data[:,col1],id_data[:,col2]
        plt.plot(x, y, color=colours[j % 32, :])
        # if i<=3:
        #     plt.plot(x,y, color=colours[j % 32, :],label='trajectory'+' '+str(j))
        # elif i==4:
        #     plt.plot(x,y, color=colours[j % 32, :],label='. . .')
        # else:
        #     plt.plot(x, y, color=colours[j % 32, :])
    plt.title("Trajectory")
    plt.ylabel("Y(m)")
    plt.xlabel("X(m)")
    # plt.legend(loc="upper left")
    plt.show()

# 绘制3d图
def draw_xy_3d(raw_txt_data):
    data=raw_txt_data
    colours = np.random.rand(32, 3)
    plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    for i in range(int(data[:, 1].max())):
        j = i + 1
        id_data = data[data[:, 1] == j, :]
        id_x = id_data[:, 2]
        id_y = id_data[:, 3]
        id_z = id_data[:, 1]

        ax.scatter3D(id_x, id_y, id_z, color=colours[j % 32, :], s=1)
    # ax.set_xticks([])
    ax.set_ylim3d(800, 1600)
    plt.title("simple 3D scatter plot")
    plt.show()

#最终数据文本
def draw_v(data,col1,col2):
    data=data
    colours = np.random.rand(32, 3)
    plt.figure(figsize=(6, 6))
    for i in range(int(data[:, 1].max())):
        j = i + 1
        id_data = data[data[:, 1] == j, :]
        time=id_data[:,col1]
        v = id_data[:, col2]
        plt.plot(time,v, color=colours[j % 32, :])
    plt.title("Velocity")
    plt.ylabel("Velocity(m/s)")
    plt.xlabel("Time(s)")
    # plt.legend()
    plt.show()

def draw_a(data,col1,col2):
    data=data
    colours = np.random.rand(32, 3)
    plt.figure(figsize=(6, 6))
    for i in range(int(data[:, 1].max())):
        j = i + 1
        id_data = data[data[:, 1] == j, :]
        time=id_data[:,col1]
        a=id_data[:,col2]
        # a=np.sqrt(a_x**2+a_y**2)
        plt.plot(time,a, color=colours[j % 32, :])
    plt.title("Acceleration")
    plt.ylabel("Accelaration(m/s\u00b2)")
    plt.xlabel("Time(s)")
    # plt.legend()
    plt.show()

def draw_headway(data,col1,col2):
    colours = np.random.rand(32, 3)
    # plt.figure(figsize=(9, 6))
    plt.figure(figsize=(12, 6))
    for i in range(int(data[:, 1].max())):
        j = i + 1
        id_data = data[data[:, 1] == j, :]
        id_data = id_data[id_data[:, 23] != 0, :]
        x,y=id_data[:,col1],id_data[:,col2]
        plt.plot(x, y, color=colours[j % 32, :])
        # if i<=3:
        #     plt.plot(x,y, color=colours[j % 32, :],label='trajectory'+' '+str(j))
        # elif i==4:
        #     plt.plot(x,y, color=colours[j % 32, :],label='. . .')
        # else:
        #     plt.plot(x, y, color=colours[j % 32, :])
    plt.title("Disdance headway")
    plt.ylabel("Disdance(m)")
    plt.xlabel("Time(s)")
    # plt.legend(loc="upper left")
    plt.show()

def draw_thw(data,col1,col2):
    colours = np.random.rand(32, 3)
    # plt.figure(figsize=(9, 6))
    plt.figure(figsize=(12, 6))
    data = data[data[:, 20] != 7, :]
    for i in range(int(data[:, 1].max())):
        j = i + 1
        id_data = data[data[:, 1] == j, :]
        id_data = id_data[id_data[:, 24] != 0, :]
        x,y=id_data[:,col1],id_data[:,col2]
        plt.plot(x, y, color=colours[j % 32, :])
        # if i<=3:
        #     plt.plot(x,y, color=colours[j % 32, :],label='trajectory'+' '+str(j))
        # elif i==4:
        #     plt.plot(x,y, color=colours[j % 32, :],label='. . .')
        # else:
        #     plt.plot(x, y, color=colours[j % 32, :])
    plt.title("Time headway")
    plt.ylabel("Time(s)")
    plt.xlabel("Time(s)")
    # plt.legend(loc="upper left")
    plt.show()

def draw_ttc(data,col1,col2):
    colours = np.random.rand(32, 3)
    # plt.figure(figsize=(9, 6))
    plt.figure(figsize=(12, 6))

    for i in range(int(data[:, 1].max())):
        j = i + 1
        id_data = data[data[:, 1] == j, :]
        id_data = id_data[id_data[:, 25] != 0, :]
        x,y=id_data[:,col1],id_data[:,col2]
        plt.plot(x, y, color=colours[j % 32, :])
        # if i<=3:
        #     plt.plot(x,y, color=colours[j % 32, :],label='trajectory'+' '+str(j))
        # elif i==4:
        #     plt.plot(x,y, color=colours[j % 32, :],label='. . .')
        # else:
        #     plt.plot(x, y, color=colours[j % 32, :])
    plt.title("Time to collision")
    plt.ylabel("Time(s)")
    plt.xlabel("Time(s)")
    # plt.legend(loc="upper left")
    plt.show()

if __name__=='__main__':
    txt_path ='output.txt'
    data = np.loadtxt(txt_path, delimiter=',',dtype=bytes).astype(str)
    data = data.astype(np.float64)
    # data =data[data[:,24]!=7,:]

    # draw_points(data,2,3)
    draw_xy(data,2,3)  #6,7
    # draw_v(data,-1,13)   #14,17
    # draw_a(data,-1,16)   #20,23
    # draw_headway(data, -1, 27)
    # draw_thw(data, -1, 28)
    # draw_ttc(data, -1, 29)
