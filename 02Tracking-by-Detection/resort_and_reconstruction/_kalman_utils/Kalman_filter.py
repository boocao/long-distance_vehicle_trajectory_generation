import os
import numpy as np
from filterpy.kalman import KalmanFilter
import random
import matplotlib.pyplot as plt
import matplotlib

# ------------------------------------
class KalmanBoxTracker(object):
    def __init__(self,point):
        # 定义恒速模型，4个状态变量，2个状态输入
        self.kf = KalmanFilter(dim_x=4,dim_z=2)
        # 状态向量 X = [点的横坐标，点的纵坐标, 点的vx速度，点的vy速度]
        # 这里假设是x和y都是匀速运动
        self.kf.F = np.array([[1,0,1,0],
                              [0,1,0,1],
                              [0,0,1,0],
                              [0,0,0,1]])
        # P是先验估计的协方差，对不可观察的初速度，给予高度不确定性。P越小说明越相信当前预测状态
        self.kf.P = np.array([[10,0,0,0],
                              [0,10,0,0],
                              [0,0,1000,0],
                              [0,0,0,1000]])
        # Q是系统状态变换误差的协方差, 一般认为系统误差很小。Q值越大，越相信测量值，
        self.kf.Q = np.array([[0.01,0,0,0],
                              [0,0.01,0,0],
                              [0,0,0.01,0],
                              [0,0,0,0.01]])
        #观测矩阵
        self.kf.H = np.array([[1,0,0,0],
                              [0,1,0,0],])
        # R是测量噪声的协方差矩阵，即真实值与测量值差的协方差。R越大，越信任估计值；越小，越信任测量值，但太小容易震荡。太小太大都不一定合适。
        self.kf.R = np.array([[10,0],
                              [0,10]])
        # Kalman滤波器初始化时，直接用第一次观测结果赋值状态信息
        self.kf.x[0:2] = point
        # 存储历史时刻的Kalman状态
        self.history = []

    def update(self,point):
        # 调用更新后，清空历史
        self.history = []
        # 直接调用包里的更新参数
        self.kf.update(point)

    def predict(self):
        # 库自带函数
        self.kf.predict()
        # 更新历史信息
        self.history.append(self.kf.x)
        return self.history[-1]

def trajectory_denoising_by_Kalman(input_x,input_y):
    x,y=input_x,input_y
    init_point = np.array([[x[0], y[0]]]).reshape(2, 1)
    pred = []
    kalman = KalmanBoxTracker(init_point)
    for i in range(x.shape[0]):
        pred.append(kalman.predict())
        # z_y = y[i] + random.gauss(0, 0.01)
        # point = np.array([[x[i], z_y]]).reshape(2, 1)
        point = np.array([[x[i], y[i]]]).reshape(2, 1)
        kalman.update(point)
    x_pred = []
    y_pred = []
    for i in range(x.shape[0]):
        x_pred.append(pred[i][0])
        y_pred.append(pred[i][1])
    return x_pred,y_pred


if __name__=='__main__':

    txt_path='./01preprocessing_txts/exp0104_preprocessing-id-100.txt'
    data = np.loadtxt(txt_path, dtype=str, delimiter=',')
    new_data = data.astype(np.float64)
    filename=os.path.basename(txt_path)

    id_100_data = new_data[new_data[:, 1] == 100, :]
    input_x,input_y = id_100_data[:,2],id_100_data[:,3]

    x_pred, y_pred=trajectory_denoising_by_Kalman(input_x,input_y)

    # 中文正常显示
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure()
    plt.plot(input_x,input_y, color='black')
    plt.plot(x_pred, y_pred, color='red')
    # plt.scatter(x, y, c='r', marker='o', label='sinx添加噪声')
    # plt.scatter(x_pred, y_pred, c='g', marker='*', label='卡尔曼滤波预测')
    plt.legend()
    plt.show()
