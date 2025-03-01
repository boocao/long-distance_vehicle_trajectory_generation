import cv2
import numpy as np
from matplotlib import pyplot as plt

# # #####手动计算转换矩阵########################
# img1 = cv2.imread('./test_0021+0123/1_stb.jpg')
# img2 = cv2.imread('./test_0021+0123/2_stb.jpg')
# # 1.先缩放右图。需要L1和L2。
# # 2.设置左图和缩放后右图的对齐点。需要P1和P2坐标。
# # 3.再旋转右图。需要旋转角。
#
# ### 缩放,x1y1为x方向缩放，x2y2为y方向缩放，
# L1, L2 = 240, 280.5 #计算右图转换到左图的缩放比例,手工数据1
# fx, fy = L1/L2, L1/L2
# M1 = np.float32([[fx, 0, 0],
#                  [0,fy, 0]])
# img2_out1 = cv2.warpAffine(img2, M1, (3840,2160))
# ### 先找一对对齐点最好，左图对齐点（3840-264,863）,缩放后的右图对齐点（178,812） #手工数据2
# X1,Y1=3840-265,812
# X2,Y2=174,782
# ##平移，x3y1向右平移，x3y2向下平移
# M2 = np.float32([[1, 0, X1-X2],
#                  [0,1, Y1-Y2]])
# print(M2)
# img2_out2 = cv2.warpAffine(img2_out1, M2, (7680,2160))
# ###沿中心旋转2.73度  #手工数据3
# delta_theta=-3.4
# M3 = cv2.getRotationMatrix2D((X1, Y1), delta_theta, 1)
# # center 旋转中心点 (cx, cy) 你可以随意指定
# # angle 旋转的角度 单位是角度 逆时针方向为正方向，角度为正值代表逆时针
# # scale 缩放倍数. 值等于1.0代表尺寸不变
# print(M3)
# img2_out3 = cv2.warpAffine(img2_out2, M3, (7680,2160))
#
# img2_out3[0:2160,0:X1] = img1[0:2160,0:X1]
# cv2.imwrite('./test_0021+0123/stitched_image1.jpg', img2_out3)
# ####################################################

img1 = cv2.imread('./exp0502/1_stb.jpg')
img2 = cv2.imread('./exp0502/2_stb.jpg')

def get_affine_mat():
    # exp0502:
    pts1=np.float32([[3416,1109],[3736,923],[3768,1267]]) ### X1, Y1为对齐点和拼接点
    pts2=np.float32([[128,1095],[508,881],[550,1283]])

    # # exp0503:
    # pts1=np.float32([[3503,1037],[3751,908],[3720,1266]]) ### X1, Y1为对齐点和拼接点
    # pts2=np.float32([[128,1058],[408,914],[381,1314]])

    # 得到变换矩阵; 进行仿射变换
    affine_mat = cv2.getAffineTransform(pts2, pts1)
    # affine_mat[0,0]=0.89

    print(affine_mat)
    return affine_mat

affine_mat=get_affine_mat()
warped_img2 = cv2.warpAffine(img2, affine_mat, (7680,2160))
cv2.imwrite('./exp0502/warped_image.jpg', warped_img2)
stitched_img=warped_img2
###分别检查三个点是否对齐
X1, Y1 = 3416,1109 # 左图对齐点位置
stitched_img[0:2160,0:X1] = img1[0:2160,0:X1]  #拼接位置在左图横坐标265处
cv2.imwrite('./exp0502/stitched_image.jpg', stitched_img)

