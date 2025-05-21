
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error,r2_score

def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    mape = (abs(y_predict - y_test) / y_test).mean()
    r_2 = r2_score(y_test, y_predict)
    return mae, rmse, mape, r_2

if __name__=='__main__':
    ####获取所要数据
    data_df = pd.read_csv('./csvs/rts_bias_acceleration.csv')   ## time,v1,v2,bias,x,y
    data=data_df.values
    # data=data[data[:,0]<65680,:]
    # data=data[data[:,0]>65680,:]
    # data=data[data[:,0]<66000,:]
    # data=data[data[:,0]>66000,:]
    # data=data[data[:,0]<66200,:]
    # data=data[data[:,0]>66200,:]

    mae, rmse, mape, r_2=evaluation(data[:,1],data[:,2])
    print(rmse)

    sns.set()  # 切换到sns的默认运行配置
    sns.distplot(data[:,3], bins=5, hist=True, kde=True, rug=False,
                 hist_kws={"color": "steelblue"}, kde_kws={"color": "purple"},
                 vertical=False, norm_hist=False)
    # 添加x轴和y轴标签
    plt.xlabel("XXX")
    plt.ylabel("YYY")

    # 添加标题
    plt.title("Distribution")
    plt.show()




