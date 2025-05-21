import cv2
import numpy as np

img = cv2.imread("./imgs/0021_0001.jpg")     #图片的路径
filename='mask_points.txt'

a = []
b = []
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "[%d,%d]" % (x, y)
        a.append(x)
        b.append(y)
        cv2.circle(img, (x, y), 3, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (107, 73, 251), thickness=1)	#如果不想在图片上显示坐标可以注释该行
        cv2.imshow("image", img)
        print("[{},{}]".format(a[-1], b[-1]))	#终端中输出坐标

        a_=np.array(a).reshape((-1,1))
        b_=np.array(b).reshape((-1,1))
        points=np.column_stack((a_,b_))
        np.savetxt(filename, points, fmt=['%0.2f', '%0.2f'], delimiter=',')

cv2.namedWindow("image",cv2.WINDOW_NORMAL) ### WINDOW_NORMAL：窗口以合适大小显示，且可以调整大小
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
