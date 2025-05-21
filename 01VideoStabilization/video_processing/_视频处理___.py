import cv2 as cv
import numpy as np
from PIL import Image
from get_masked_image import image_process

def main():
    video_path = "F:/AerialVideos/04Trajectory/exp0503/stitched_video.mp4"
    out_path = "F:/AerialVideos/04Trajectory/exp0503/stitched_video_test.mp4"

    video_caputre = cv.VideoCapture(video_path)
    FPS = video_caputre.get(cv.CAP_PROP_FPS)
    width = video_caputre.get(cv.CAP_PROP_FRAME_WIDTH)
    height = video_caputre.get(cv.CAP_PROP_FRAME_HEIGHT)

    size = (int(3840), int(2160))#先宽后高
    video_writer = cv.VideoWriter(out_path, cv.VideoWriter_fourcc('m', 'p', '4', 'v'), FPS, size)

    n = 0
    while True:
        success, frame = video_caputre.read()
        frame_target = frame[0:2160, 1600:5440]
        #截取全部或某一段
        if success:
            n += 1
            print(n)
            if n<9999:
                video_writer.write(frame_target)
            else:
                break
        else:
            break

if __name__=="__main__":
    main()