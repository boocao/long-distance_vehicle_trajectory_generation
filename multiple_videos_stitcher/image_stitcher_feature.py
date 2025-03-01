import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('./exp0503/1_stb.jpg')
img2 = cv2.imread('./exp0503/2_stb.jpg')

###########################################################
####第一种常规方法，可行，需要试出好的H
# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img1_tailor=img1.copy()
img2_tailor=img2.copy()
img1_p1x,img1_p1y= 3394,898
img1_p2x,img1_p2y= 3840,1284
img2_p1x,img2_p1y= 0,897
img2_p2x,img2_p2y=459,1335
# img1_tailor[0:2160,0:img1_p1x]=0
# img1_tailor[0:img1_p1y,0:3840]=0
# img1_tailor[img1_p2y:2160,0:3840]=0
# img2_tailor[0:2160,img2_p2x:3840]=0
# img2_tailor[0:img2_p1y,0:3840]=0
# img2_tailor[img2_p2y:2160,0:3840]=0

#################################################################
##特征点检测
sift = cv2.xfeatures2d.SIFT_create()
# sift = cv2.xfeatures2d.SURF_create()
## keypoints, descriptor = surf.detectAndCompute(img, None)
psd_kp1, psd_des1 = sift.detectAndCompute(img1_tailor, None)
psd_kp2, psd_des2 = sift.detectAndCompute(img2_tailor, None)
# img = cv2.drawKeypoints(image=img2,
#                         outImage=img2,
#                         keypoints = psd_kp2,
#                         flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
#                         color = (0, 0, 255))

# ##匹配器1：BF特征特征匹配器
# bf = cv2.BFMatcher()
# #使用描述子进行一对多的描述子匹配
# matches = bf.knnMatch(psd_des1,psd_des2,k=2)

##匹配器2：Flann特征匹配器
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # 指定索引树要被遍历的次数
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(psd_des1, psd_des2, k=2)

#筛选有效的特征描述子存入数组中
verify_matches = []
for m1,m2 in matches:
    if m1.distance <0.5*m2.distance:
        verify_matches.append(m1)
goodMatch = np.expand_dims(verify_matches, 1)
img_out_match=cv2.drawMatchesKnn(img1, psd_kp1,
                           img2, psd_kp2,
                           goodMatch[:], None,[255,0,0], flags=2)

##单应性矩阵需要最低四个特征描述子坐标点进行计算，判断数组中是否有足够,这里设为6更加充足
if len(verify_matches) > 6:
    ##存放求取单应性矩阵所需的img1和img2的特征描述子坐标点
    img1_pts = []
    img2_pts = []
    for m in verify_matches:
    ##通过使用循环获取img1和img2图像优化后描述子所对应的特征点
        img1_pts.append(psd_kp1[m.queryIdx].pt)
        img2_pts.append(psd_kp2[m.trainIdx].pt)
    ##得到的坐标是[(x1,y1),(x2,y2),....]
    ##计算需要的坐标格式：[[x1,y1],[x2,y2],....]所以需要转化
    img1_pts = np.float32(img1_pts).reshape(-1,1,2)
    img2_pts = np.float32(img2_pts).reshape(-1,1,2)
###########################################################################
    ##计算单应性矩阵
    H,mask = cv2.findHomography(img2_pts,img1_pts,cv2.RANSAC,5.0)
    print(H)
    result_img = cv2.warpPerspective(img2,H,(7680,2160))
    result_img[:2160, :(3840-252)] = img1[:2160, :(3840-252)]
    cv2.imwrite('./exp0503/stitched_image3.jpg', result_img)
    ##CitySIM重叠像素估计有2000多，2000效果应该会更好。
###############################################################################################
###############################################################################################
# ####第二种方法可选，实际没啥用，选出比较好的3个点计算仿射变换矩阵
#     # 只利用4对点计算单应性矩阵:
#     def Homographyfrom4Pts(pair_points):
#         pt1 = pair_points[:, :2].astype(np.float32)
#         pt2 = pair_points[:, 2:4].astype(np.float32)
#         # 可能的问题出在三点共线或者两点重合的情况，导致误差巨大
#         M = cv2.getPerspectiveTransform(pt1, pt2)
#         # 返回的行列式用于辅助检查 M 是否正确
#         return M, np.linalg.det(M)
#
#     def RANSAC(match_pts):
#         pts_num = match_pts.shape[0]
#         det_M = 0
#         update_match_pts = []
#         max_satisfy_rate = 0
#         # 最大迭代次数100
#         for i in range(100):
#             det_M = 0
#             while (det_M <= 0.1):
#                 # 随机选取4对点
#                 rand4 = np.random.randint(pts_num, size=4)
#                 # 基于这4对点计算单应性矩阵
#                 # print(match_pts[rand4, :])
#                 M, det_M = Homographyfrom4Pts(match_pts[rand4, :])
#
#             # 添加齐次坐标
#             homo_pts1 = np.insert(match_pts[:, :2], 2, values=np.ones((1, pts_num)), axis=1).T
#             # 重投影齐次坐标
#             homo_pts2_hat = (M @ homo_pts1).T
#             # 重投影坐标
#             pts2_hat = (homo_pts2_hat / homo_pts2_hat[:, 2].reshape(-1, 1))[:, :2]
#             # 计算误差
#             error_matrix = np.sum((match_pts[:, 2:4] - pts2_hat) ** 2, axis=1)
#             satisfy_rate = sum(error_matrix < 10) / pts_num
#             # 若重投影正确率大于当前最大值, 更新认为是正确的匹配点
#             if (satisfy_rate > max_satisfy_rate):
#                 max_satisfy_rate = satisfy_rate
#                 update_match_pts = match_pts[error_matrix < 10]
#             # 若重投影正确率大于阈值, 直接返回结果
#             if (satisfy_rate > 0.75):
#                 return update_match_pts
#         return update_match_pts
#
#     match_pts=np.column_stack((img2_pts.reshape(-1,2),img1_pts.reshape(-1,2)))
#     update_match_pts=RANSAC(match_pts)
#
#     img2_pts_=update_match_pts[:,:2].astype(np.float32)
#     img1_pts_=update_match_pts[:,2:4].astype(np.float32)
#     plt.scatter(img1_pts_[:,0],img1_pts_[:,1],c='blue',)
#     # plt.scatter(img2_pts_[:,0],img2_pts_[:,1],c='red')
#     for i in range(len(img1_pts_)):
#         plt.text(img1_pts_[i,0], img1_pts_[i,1], (i))
#     plt.imshow(img1)
#     plt.show()
#
#     pts1=np.row_stack((img1_pts_[0,:],img1_pts_[2,:],img1_pts_[5,:]))
#     pts2=np.row_stack((img2_pts_[0,:],img2_pts_[2,:],img2_pts_[5,:]))
#     affine_mat = cv2.getAffineTransform(pts2, pts1)
#     img_out = cv2.warpAffine(img2, affine_mat, (7680,2160))
#     img_out[:2160,:3840] = img1
#
#     # # 可以去除黑色无用部分
#     # rows, cols = np.where(img_out[:, :, 0] != 0)
#     # min_row, max_row = min(rows), max(rows) + 1
#     # min_col, max_col = min(cols), max(cols) + 1
#     # result = img_out[min_row:max_row, min_col:max_col, :]
#     cv2.imwrite('./exp0501/stitched_image3.jpg', img_out)