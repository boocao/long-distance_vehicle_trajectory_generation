import cv2
import numpy as np

def image_process(img, points):
    points = np.array([points], dtype=np.int32)
    ###绘制mask
    zeros = np.zeros((img.shape), dtype=np.uint8)
    color_light_yellow=(0, 165, 255)   ##浅黄色
    color_light_green=(144, 238, 144)  ##浅绿色
    # # 原本thickness = -1表示内部填充,这里不知道为什么会报错,只好不填充了 改用函数cv2.polylines
    # cv2.polylines(img, points, isClosed=True, thickness=5, color=(144, 238, 144))
    mask = cv2.fillPoly(zeros, points, color=color_light_yellow)  ####填充颜色

    ##绘制轮廓
    # img=cv2.drawContours(img, points, -1, (144, 238, 144), 5)  ###绘制轮廓

    ##叠加mask和普通图片
    masked1 = 0.3 * mask + img

    mask=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    masked2 = cv2.bitwise_and(img, img, mask=mask)
    return masked2
if __name__=='__main__':
    img = cv2.imread(r'./imgs/0021_0001.jpg')
    filename='mask_points.txt'
    mask_points = np.loadtxt(filename, delimiter=',',dtype=bytes ).astype(str)
    mask_points = mask_points.astype(np.float64)
    mask_image=image_process(img=img, points=mask_points)
    cv2.imwrite('masked_image.jpg', mask_image)