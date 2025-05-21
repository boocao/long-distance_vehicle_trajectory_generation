
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from PIL import Image


if __name__=='__main__':
    x1 = np.arange(-2, 1.1, 0.1)
    y1=x1+1
    x2 = np.arange(-1, 2.1, 0.1)
    y2=x2-1
    x3 = np.arange(-1, 1.1, 0.1)
    # y3=0*x3
    y3=(x3*x3*x3-x3)/2
    x4 = np.arange(-2, -0.9, 0.1)
    y4=x4+1
    x5 = np.arange(1, 2.1, 0.1)
    y5=x5-1

    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
    plt.plot(x1, y1, ".",markersize=3,linestyle='-',linewidth=3,color='gray',)
    plt.plot(x2, y2, ".",markersize=3,linestyle='-',linewidth=3,color='gray',)
    plt.plot(x3, y3, ".",markersize=3,linestyle='-',linewidth=4,color='black',)
    plt.plot(x4, y4, ".",markersize=3,linestyle='-',linewidth=4,color='black')
    plt.plot(x5, y5, ".",markersize=3,linestyle='-',linewidth=4,color='black')
    ax = plt.gca()  # 获取整张图像的坐标的对象
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    plt.xticks(fontproperties='Times New Roman',size=20)
    plt.yticks([-2,-1.5,-1,-0.5,0.5,1,1.5,2],fontproperties='Times New Roman',size=20)
    # plt.legend(fontsize=18,loc="upper left")
    # plt.show()
    plt.savefig('test.png', format='png', transparent=True, dpi=900)

    

