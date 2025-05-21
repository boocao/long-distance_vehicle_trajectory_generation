import cv2 as cv
import numpy as np
from PIL import Image
from get_masked_image import image_process

def main():
    video_path = 'E:/07Experiment-data/VideoSets/02AerialVideos-02稳像后视频/Stitching202310261200-H220-4K30fps_1-right_199+200+201_stb_cutted_wraped.mp4'
    # video_path = "E:/07Experiment-data/VideoSets/02AerialVideos-03蒙版的视频/exp0201_masked.mp4"
    out_path = "E:/07Experiment-data/VideoSets/02AerialVideos-03蒙版的视频/exp0502-right_masked.mp4"

    video_caputre = cv.VideoCapture(video_path)
    FPS = video_caputre.get(cv.CAP_PROP_FPS)
    width = video_caputre.get(cv.CAP_PROP_FRAME_WIDTH)
    height = video_caputre.get(cv.CAP_PROP_FRAME_HEIGHT)

    size = (int(width), int(height))#先宽后高
    new_FPS = FPS #调整帧率
    video_writer = cv.VideoWriter(out_path, cv.VideoWriter_fourcc('m', 'p', '4', 'v'), new_FPS, size)

    n = 0
    while True:
        success, frame = video_caputre.read()
        #截取全部或某一段
        if success:
            n += 1
            print(n)
            if n>=0:

                mask_image = image_process(img=frame, points=mask_points)
                video_writer.write(mask_image)
            else:
                break
        else:
            break

filename='mask_points.txt'
mask_points = np.loadtxt(filename, delimiter=',',dtype=bytes).astype(str)
mask_points = mask_points.astype(np.float64)

if __name__=="__main__":
    main()