import os.path
import numpy as np
import cv2
import argparse

# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")
def draw_mot(frame,per_frame_data,existed_data):
    draw_frame = frame.copy()
    for i in range(int(per_frame_data[:, 1].max())):
        per_id_data = per_frame_data[per_frame_data[:, 1] == (i + 1), :]
        if per_id_data.shape[0] <= 0:
            continue
        box = per_id_data[0, 2:6]
        box[0:2] = per_id_data[0, 6:8]
        id = per_id_data[0, 1]
        cls = per_id_data[0, 8]

        color = [int(c) for c in COLORS[int(id) % len(COLORS)]]
        color_tuple = (color[0], color[1], color[2])
        color_light_green = (144, 238, 144)  ##浅绿色

        ###先画透明框
        cv2.rectangle(draw_frame, (int(float(box[0] - box[2] / 2)), int(float(box[1] - box[3] / 2))),
                      (int(float(box[0] + box[2] / 2)), int(float(box[1] + box[3] / 2))),
                      color=color_light_green, thickness=-1)
        ### 绘制轨迹线
        line_data = existed_data[existed_data[:, 1] == id, :]
        line_data = line_data[np.argsort(line_data[:, 0]), :]
        if len(line_data) > 120:
            line_data = line_data[-120:, :]
        for i, num in enumerate(line_data):
            if i == 0:
                continue
            else:
                cv2.line(draw_frame, tuple(map(int, line_data[i, 2:4])), tuple(map(int, line_data[i - 1, 2:4])),
                         color_light_green, 2, 8)
    alpha = 0.5
    draw_frame = cv2.addWeighted(frame, alpha, draw_frame, 1 - alpha, gamma= 0)

    for i in range(int(per_frame_data[:, 1].max())):
        per_id_data = per_frame_data[per_frame_data[:, 1] == (i + 1), :]
        if per_id_data.shape[0] <= 0:
            continue
        box = per_id_data[0, 2:6]
        box[0:2] = per_id_data[0, 6:8]
        id = per_id_data[0, 1]
        cls = per_id_data[0, 8]

        color = [int(c) for c in COLORS[int(id) % len(COLORS)]]
        color_tuple = (color[0], color[1], color[2])
        color_light_green = (144, 238, 144)  ##浅绿色
        Sky_blue_grey = (202, 235, 216)  ##天蓝灰
        Ivory_black = (41,36,33)

        ###画边框
        cv2.rectangle(draw_frame, (int(float(box[0] - box[2] / 2)), int(float(box[1] - box[3] / 2))),
                      (int(float(box[0] + box[2] / 2)), int(float(box[1] + box[3] / 2))),
                      color=color_light_green, thickness=3)
        ## 填写标签部分
        label = ['car', 'truck', 'bus'][int(cls)]
        labelSize = cv2.getTextSize(label + str(int(id)), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        # labelSize = cv2.getTextSize('ID' + str(int(id)), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        # print(labelSize)
        cv2.rectangle(draw_frame, (int(float(box[0] - box[2] / 2)), int(float(box[1] - box[3] / 2 - labelSize[1]))),
                      (int(float(box[0] - box[2] / 2 + labelSize[0]-5)), int(float(box[1] - box[3] / 2))),
                      color=Sky_blue_grey, thickness=-1)
        cv2.putText(draw_frame, label + ' ' + str(int(id)),
                    (int(float(box[0] - box[2] / 2 + 4)), int(float(box[1] - box[3] / 2 - 3))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, Ivory_black, thickness=2)
    return draw_frame

if __name__ == '__main__':
    path_txt = "E:/07Experiment-data/VideoSets/03AerialVideos-LongTrajectoryFiles/exp0502/exp0502test_reconstructed.txt"
    path_video = "E:/07Experiment-data/VideoSets/03AerialVideos-LongTrajectoryFiles/exp0502/stitched_video.mp4"
    output_path = 'E:/07Experiment-data/VideoSets/03AerialVideos-LongTrajectoryFiles/exp0502/exp0502_visualizationd.mp4'
    # output_path = "E:/07Experiment-data/VideoSets/02AerialVideos-04TextFiles/03merge/exp0303/exp0303_visualization_part.mp4"

    cap = cv2.VideoCapture(path_video)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率<每秒中展示多少张图片>
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取高度
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height), True)

    data = np.loadtxt(path_txt, delimiter=',',dtype=bytes).astype(str)
    data = data.astype(np.float64)

    ###绘图所用数据格式为转换后格式，即frames, ids, centers_x, centers_y, widths, heights,class
    print('draw_mot')
    n = 0
    while True:
        success, frame = cap.read()
        if success:
            if n<=999999:
                n += 1
                print(n)

                per_frame_data = data[data[:, 0] == n, :]
                existed_data = data[data[:, 0] <= n, :]
                if per_frame_data.shape[0] <= 0:
                    frame_out=frame
                else:
                    frame_out = draw_mot(frame,per_frame_data,existed_data)

                out.write(frame_out)
            else:
                break
        else:
            break

