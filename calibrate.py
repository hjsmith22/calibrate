# general libraries
import numpy as np                        # arrays and math functions
import pandas as pd                       # data manipulation and analysis
import cv2                                # OpenCV: computer vision and machine learning
import os                                 # creating and changing directories
import matplotlib.pyplot as plt           # makes python work like MATLAB: plotting and graphing
import matplotlib as mpl
import crop_videos as crop                # crop_videos.py
from mpl_toolkits.mplot3d import Axes3D   # 3d plots
import PIL                                # opening, manipulating, and saving images
import glob                               # search files
import pickle                             # serialization of more advanced data types

# charuco libraries
from cv2 import aruco
# matplotlib nbagg ???

# cpp charuco libraries ?
#include <opencv2/highgui.hpp>
#include "aruco_samples_utility.hpp"

# anipose calibration libraries (might not need in this script)
''' from tqdm import tqdm
import numpy as np

from glob import glob
from collections import defaultdict '''

'''
# AVI class
class Video(object):
    __init__:(self, path)
    self.path = path
# need array of frames or something like that
class AVI(Video):
    type = "MP4"
'''

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def make_board()
    aruco_type = "DICT_5X5_250"
    id = 1

    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])

    print("ArUCo type '{}' with ID '{}'".format(aruco_type, id))
    tag_size = 250
    tag = np.zeros((tag_size, tag_size, 1), dtype="uint8")
    cv2.aruco.drawMarker(arucoDict, id, tag_size, tag, 1)
    
    # Save the tag g
    # enerated
    tag_name = "arucoMarkers/" + aruco_type + "_" + str(id) + ".png"
    cv2.imwrite(tag_name, tag)
    cv2.imshow("ArUCo Tag", tag)

    cv2.waitKey(0)

    cv2.destroyAllWindows()


# line 41 to go in main, probably
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


# load video and store frames
folders = find_calibration_vid_folder('''dont know what file goes here''')

def load_video(file)
    # load video

# pickle the video into frames
def make_frames('''video'''):
    # open avi
    video = cv2.VideoCapture('''avi directory''') # this might be skilled reaching videos
    # make directory for frames
    os.makedirs('''frames directory''')
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
        filename = os.path.join('''frames directory''', f'frame{count:04d}.png')
        cv2.imwrite(filename, frame)
        # add frame to array
        frames.append(frame)
        count += 1
    # release video
    video.release()
# now should have "frames" array

# take photos from multiple angles EXAMPLE
# this is probably where each frame of the video comes in
datadir = "../../data/calib_tel_ludo/" # C:\Users\dleventh\Dropbox (University of Michigan)\MED-LeventhalLab\data\dLight_photometry\calibration_videos
images = np.array([datadir + f for f in os.listdir(datadir) if f.endswith(".png") ]) # images might not be pngs, change later if theyre not
order = np.argsort([int(p.split(".")[-2].split("_")[-1]) for p in images])
images = images[order]
# images

im = PIL.Image.open(images[0])
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.imshow(im)
ax.axis('off')
plt.show()

# detect markers on the images
def read_board(frames):
    """
    Charuco base pose estimation.
    """
    print("POSE ESTIMATION STARTS:")
    allCorners = []
    allIds = []
    decimator = 0
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    for im in frames:
        print("=> Processing image {0}".format(im))
        frame = cv2.imread(im)
        # converts frame from color to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detects markers on the charuco board
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)
        # draw detected markers on image
        cv2.aruco.drawDetectedMarkers(im, corners, ids)
        '''# interpolates corners
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, parameters)'''
        # Draw the interpolated corners on the image
        cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)
        # Display the image
        cv2.imshow('Charuco Board', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if len(corners)>0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize = (3,3),
                                 zeroZone = (-1,-1),
                                 criteria = criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,gray,board)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
                allCorners.append(res2[1])
                allIds.append(res2[2])

        decimator+=1

    imsize = gray.shape
    return allCorners,allIds,imsize

allCorners,allIds,imsize=read_chessboards(images)

# The second will proceed the detected markers to estimage the camera calibration data.
def calibrate_camera(allCorners,allIds,imsize):
    """
    Calibrates the camera using the dected corners.
    """
    print("CAMERA CALIBRATION")

    cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
                                 [    0., 1000., imsize[1]/2.],
                                 [    0.,    0.,           1.]])

    distCoeffsInit = np.zeros((5,1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    #flags = (cv2.CALIB_RATIONAL_MODEL)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=allCorners,
                      charucoIds=allIds,
                      board=board,
                      imageSize=imsize,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors

# %time ret, mtx, dist, rvecs, tvecs = calibrate_camera(allCorners,allIds,imsize)
'''
CAMERA CALIBRATION
CPU times: user 10.3 s, sys: 8.89 s, total: 19.2 s
Wall time: 5.26 s

ret

# check cailibration results
i=20 # select image id
plt.figure()
frame = cv2.imread(images[i])
img_undist = cv2.undistort(frame,mtx,dist,None)
plt.subplot(1,2,1)
plt.imshow(frame)
plt.title("Raw image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(img_undist)
plt.title("Corrected image")
plt.axis("off")
plt.show()

# use of calibration to estimate 3d translation and rotation aof each marker on a scene
frame = cv2.imread("../../data/IMG_20180406_095219.jpg")
#frame = cv2.undistort(src = frame, cameraMatrix = mtx, distCoeffs = dist)
plt.figure()
plt.imshow(frame, interpolation = "nearest")
plt.show()
'''

import numpy as np
from aniposelib.boards import CharucoBoard, Checkerboard
from aniposelib.cameras import Camera, CameraGroup
from aniposelib.utils import load_pose2d_fnames

vidnames = [['calib-charuco-camA-compressed.MOV'],
            ['calib-charuco-camB-compressed.MOV'],
            ['calib-charuco-camC-compressed.MOV']]

cam_names = ['A', 'B', 'C']

n_cams = len(vidnames)

board = CharucoBoard(7, 10,
                     square_length=25, # here, in mm but any unit works
                     marker_length=18.75,
                     marker_bits=4, dict_size=50)


# the videos provided are fisheye, so we need the fisheye option
cgroup = CameraGroup.from_names(cam_names, fisheye=True)

# this will take about 15 minutes (mostly due to detection)
# it will detect the charuco board in the videos,
# then calibrate the cameras based on the detections, using iterative bundle adjustment
cgroup.calibrate_videos(vidnames, board)

# if you need to save and load
# example saving and loading for later
cgroup.dump('calibration.toml')

## example of loading calibration from a file
## you can also load the provided file if you don't want to wait 15 minutes
cgroup = CameraGroup.load('calibration.toml')

## example triangulation without filtering, should take < 15 seconds
fname_dict = {
    'A': '2019-08-02-vid01-camA.h5',
    'B': '2019-08-02-vid01-camB.h5',
    'C': '2019-08-02-vid01-camC.h5',
}

d = load_pose2d_fnames(fname_dict, cam_names=cgroup.get_names())

score_threshold = 0.5

n_cams, n_points, n_joints, _ = d['points'].shape
points = d['points']
scores = d['scores']

bodyparts = d['bodyparts']

# remove points that are below threshold
points[scores < score_threshold] = np.nan

points_flat = points.reshape(n_cams, -1, 2)
scores_flat = scores.reshape(n_cams, -1)

p3ds_flat = cgroup.triangulate(points_flat, progress=True)
reprojerr_flat = cgroup.reprojection_error(p3ds_flat, points_flat, mean=True)

p3ds = p3ds_flat.reshape(n_points, n_joints, 3)
reprojerr = reprojerr_flat.reshape(n_points, n_joints)

# plot the x, y, z coordinates of joint 0

import matplotlib.pyplot as plt
# % matplotlib notebook

plt.figure(figsize=(9.4, 6))
plt.plot(p3ds[:, 0, 0])
plt.plot(p3ds[:, 0, 1])
plt.plot(p3ds[:, 0, 2])
plt.xlabel("Time (frames)")
plt.ylabel("Coordinate (mm)")
plt.title("x, y, z coordinates of {}".format(bodyparts[0]))

## plot the first frame in 3D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import get_cmap
# %matplotlib notebook

def connect(ax, points, bps, bp_dict, color):
    ixs = [bp_dict[bp] for bp in bps]
    return ax.plot(points[ixs, 0], points[ixs, 1], points[ixs, 2], color=color)

def connect_all(ax, points, scheme, bodyparts, cmap=None):
    if cmap is None:
        cmap = get_cmap('tab10')
    bp_dict = dict(zip(bodyparts, range(len(bodyparts))))
    lines = []
    for i, bps in enumerate(scheme):
        line = connect(ax, points, bps, bp_dict, color=cmap(i)[:3])
        lines.append(line)
    return lines

## scheme for the hand
scheme = [
   ["MCP1", "PIP1", "tip1"],
   ["MCP2", "PIP2", "DIP2", "tip2"],
   ["MCP3", "PIP3", "DIP3", "tip3"],
   ["MCP4", "PIP4", "DIP4", "tip4"],
   ["MCP5", "PIP5", "DIP5", "tip5"],
   ["base", "MCP1", "MCP2", "MCP3", "MCP4", "MCP5", "base"]
 ]

framenum = 0
p3d = p3ds[framenum]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(p3d[:,0], p3d[:,1], p3d[:,2], c='black', s=100)
connect_all(ax, p3d, scheme, bodyparts)
