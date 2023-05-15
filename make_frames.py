def make_frames(video):
    # open avi
    video = cv2.VideoCapture(video_directory) # this might be skilled reaching videos
    # make directory for frames
    os.makedirs(frames_directory)
    # make sure it's an avi
    if not video.lower().endswith('.avi'):
        exit()
    # initialize frames array
    frames = []
    # loop through every frame of the video
    count = 0
    while True:
        # read frame
        ret, frame = video.read()
        # check if frame was successfully read
        if not ret:
            break
        # save frame as pngs
        filename = os.path.join(frames_directory, f'frame{count:04d}.png')
        cv2.imwrite(filename, frame)
        # add frame to array
        frames.append(frame)
        count += 1
    # release video
    video.release()
# now should have "frames" array

directory = "./directory/" # /Users/deleventh/Desktop/hayden_anipose/skilled_reaching/calibrate.py
aruco_dictionary = aruco.Dictionary_get(aruco.DICT_6X6_250)
board = aruco.CharucoBoard_create(7, 5, 1, .8, aruco_dictionary)
board_image = board.draw((2000, 2000))
cv2.imwrite(directory + "chessboard.tiff", board_image)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.imshow(board_image, cmap = mpl.cm.gray, interpolation = "nearest")
ax.axis("off")
plt.show()
