import cv2 as cv
import os
def main():
    # video_path ="E:/07Experiment-data/VideoSets/02AerialVideos-02稳像后视频/Stitching202310261200-H220-4K30fps_1-right_199+200+201_stb_cutted.mp4"
    video_path ="F:/AerialVideos/04Trajectory/exp0503/stitched_video_test.mp4"
    out_path = "./imgs/"
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    video_caputre = cv.VideoCapture(video_path)
    fps = video_caputre.get(cv.CAP_PROP_FPS)
    width = video_caputre.get(cv.CAP_PROP_FRAME_WIDTH)
    height = video_caputre.get(cv.CAP_PROP_FRAME_HEIGHT)

    size = (int(width), int(height))#先宽后高
    new_fps = fps #调整帧率
    video_writer = cv.VideoWriter(out_path, cv.VideoWriter_fourcc('m', 'p', '4', 'v'), new_fps, size)

    n = 0
    while True:
        success, frame = video_caputre.read()
        # frame_target = frame_src[0:int(width/2), 0:int(height)]
        if success:
            #截取全部或某一段
            n += 1
            print(n)
            if n<=7200:
                continue
            if n%1==0:
                # video_writer.write(frame)
                # name=out_path +'15-'+ str(n) +'.jpg'
                name = os.path.join(os.path.abspath(out_path), '0021_' + format(str(n), '0>4s') + '.jpg')
                cv.imwrite(name,frame)

        else:
            break

if __name__=="__main__":
    main()