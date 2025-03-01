import cv2

out_path = "./video_combination1.mp4"
out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (3840,2160))

videopath1='E:/07Experiment-data/VideoSets/02AerialVideos/202310261200-H220-4K30fps-multi_videos/DJI_0206.MP4'
videopath2='E:/07Experiment-data/VideoSets/02AerialVideos/202310261200-H220-4K30fps-multi_videos/DJI_0207.MP4'
videopath3='E:/07Experiment-data/VideoSets/02AerialVideos/202310261200-H220-4K30fps-multi_videos/DJI_0208.MP4'
videopath4='E:/07Experiment-data/VideoSets/02AerialVideos/202310261200-H220-4K30fps-multi_videos/DJI_0209.MP4'
# videopath5='E:\\202208041800-H240+200-4K30fps\\DJI_0069.MP4'
for path in [videopath1, videopath2, videopath3,videopath4 ]:  # 需要合并的视频名称，也可以用os.listdir()
    # video_caputre = cv2.VideoCapture(path)
    # fps = video_caputre.get(cv2.CAP_PROP_FPS)
    # width = video_caputre.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = video_caputre.get(cv2.CAP_PROP_FRAME_HEIGHT)

    cap = cv2.VideoCapture(path)
    # frameToStart = 100
    # cap.set(cv2.CAP_PROP_POS_FRAMES, frameToStart)
    n=0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # cv2.imshow('frame', frame)
        # ch = 0xFF & cv2.waitKey(30)
        n=n+1
        print(n)
        out.write(frame)
        # if ch == 27:
        #     break
out.release()
cv2.destroyAllWindows()







# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# out = cv2.VideoWriter('merged_drum.avi',fourcc, 25.0, (1920, 1080))
#
# for path in ['new1.avi','new4.avi','new2.avi']: # 需要合并的视频名称，也可以用os.listdir()
#     cap = cv2.VideoCapture(path)
#     # frameToStart = 100
#     # cap.set(cv2.CAP_PROP_POS_FRAMES, frameToStart)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         cv2.imshow('frame', frame)
#         ch = 0xFF & cv2.waitKey(30)
#         out.write(frame)
#         if ch == 27:
#             break
#     cap.release()  # 关闭相机
# out.release()
# cv2.destroyAllWindows()
