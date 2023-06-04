import numpy as np
import cv2
import os
from aniposelib.boards import CharucoBoard, Checkerboard
from aniposelib.cameras import Camera, CameraGroup
from aniposelib.utils import load_pose2d_fnames
import matplotlib.pyplot as plt
# % matplotlib notebook
from matplotlib.pyplot import get_cmap
# %matplotlib notebook

import glob
import os
# from moviepy.editor import *
import subprocess
import cv2
import shutil
import pandas as pd
from datetime import datetime
from tqdm import tqdm

import skilled_reaching_calibration
import navigation_utilities

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

def crop_video(vid_path_in, vid_path_out, crop_params, view_name, filtertype='mjpeg2jpeg'):

    # crop videos losslessly. Note that the trick of converting the video into a series of jpegs, cropping them, and
    # re-encoding is a trick that only works because our videos are encoded as mjpegs (which apparently is an old format)

    x1, x2, y1, y2 = [int(cp) for cp in crop_params]
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    vid_root, vid_name = os.path.split(vid_path_out)

    print('cropping {}'.format(vid_name))

    if filtertype == 'mjpeg2jpeg':
        jpg_temp_folder = os.path.join(vid_root, 'temp')

        # if path already exists, delete the old temp folder. Either way, make a new one.
        if os.path.isdir(jpg_temp_folder):
            shutil.rmtree(jpg_temp_folder)
        os.mkdir(jpg_temp_folder)

        full_jpg_path = os.path.join(jpg_temp_folder, 'frame_%d.jpg')
        # full_jpg_crop_path = os.path.join(jpg_temp_folder, 'frame_crop_%d.jpg')
        command = (
            f"ffmpeg -i {vid_path_in} "
            f"-c:v copy -bsf:v mjpeg2jpeg {full_jpg_path} "
        )
        subprocess.call(command, shell=True)

        # find the list of jpg frames that were just made, crop them, and resave them
        jpg_list = glob.glob(os.path.join(jpg_temp_folder, '*.jpg'))
        for jpg_name in tqdm(jpg_list):
            img = cv2.imread(jpg_name)
            cropped_img = img[y1-1:y2-1, x1-1:x2-1, :]
            if view_name == 'rightmirror':
                # flip the image left to right so it can be run through a single "side mirror" DLC network
                cropped_img = cv2.flip(cropped_img, 1)   # 2nd argument flipCode > 0 indicates flip horizontally
            cv2.imwrite(jpg_name, cropped_img)

        # turn the cropped jpegs into a new movie
        command = (
            f"ffmpeg -i {full_jpg_path} "
            f"-c:v copy {vid_path_out}"
        )
        subprocess.call(command, shell=True)

        # destroy the temp jpeg folder
        shutil.rmtree(jpg_temp_folder)
    elif filtertype == 'h264':
        command = (
            f"ffmpeg -n -i {vid_path_in} "
            f"-filter:v crop={w}:{h}:{x1}:{y1} "
            f"-c:v h264 -c:a copy {vid_path_out}"
        )
        subprocess.call(command, shell=True)

if __name__ == '__main__':

    # vid_path = r'C:\Users\haydenjs\Desktop\overhead_calibration_videos'
    vid_path = r'C:\Users\haydenjs\Desktop'
    # vid_path_out = r'C:\Users\haydenjs\Desktop'
    # vid_path = r'C:\Users\haydenjs\Documents\hand-demo\2019-08-02\calibration\bak-charuco'

    vidnames = [['cam00_sr_overhead.mov'],
                ['cam01_sr_overhead.mov']]

    # anipose
    # vidnames = [['calib-charuco-camA.MOV'],
                # ['calib-charuco-camB.MOV'],
                # ['calib-charuco-camC.MOV']]

    # skilled reaching front
    # vid_path = 'C:\Users\haydenjs\Desktop
    # nothing

    # skilled reaching overhead (R0452)
    # vid_path = 'C:\Users\haydenjs\Desktop
    # cam 00, 'cam00_sr_overhead.mov', 0/199
    # cam 01, 'cam01_sr_overhead.mov', 0/230
    # cam 02, 'cam02_sr_overhead.mov', 2/234
    # cam 03, 'cam03_sr_overhead.mov', 25/218

    # pavlovian overhead
    # vid_path = 'C:\Users\haydenjs\Desktop\overhead_calibration_videos'
    # cam 00, 'cam00_pav_overhead.mov', 93/195
    # cam 01, 'cam01_pav_overhead.mov', 3/204
    # cam 02, 'cam02_pav_overhead.mov', 141/180
    # cam 03, 'cam03_pav_overhead.mov', 160/170



    vidnames = [[os.path.join(vid_path, vn[0])] for vn in vidnames]

    # vidnames_out = [['GridCalibration_box01_20230313_12-55-56_cropped.avi'],
                    # ['GridCalibration_box01_20230313_12-55-56_cropped.avi'],
                    # ['GridCalibration_box01_20230313_12-55-56_cropped.avi']]

    # vidnames_out = [[os.path.join(vid_path_out, vn[0])] for vn in vidnames_out]

    crop_params = [680,  # left
                   1360,  # right
                   0,  # top
                   1080  # bottom
                   ]

    # crop_video(vidnames[0][0], vidnames[0][0], crop_params, view_name="direct", filtertype='mjpeg2jpeg')
    # crop_video(vidnames[1][0], vidnames[1][0], crop_params, view_name="direct", filtertype='mjpeg2jpeg')
    # crop_video(vidnames[2][0], vidnames[2][0], crop_params, view_name="direct", filtertype='mjpeg2jpeg')

    cam_names = ['A', 'B', 'C']

    n_cams = len(vidnames)

    board = CharucoBoard(7, 10,
                         square_length=25,  # here, in mm but any unit works
                         marker_length=18.75,
                         marker_bits=4, dict_size=50)

    # the videos provided are fisheye, so we need the fisheye option
    cgroup = CameraGroup.from_names(cam_names, fisheye=True)

    # this will take about 15 minutes (mostly due to detection)
    # it will detect the charuco board in the videos,
    # then calibrate the cameras based on the detections, using iterative bundle adjustment
    cgroup.calibrate_videos(vidnames, board)
    # cgroup.calibrate_videos(vidnames, board)

    # if you need to save and load
    # example saving and loading for later
    # cgroup.dump('calibration.toml')

    ## example of loading calibration from a file
    ## you can also load the provided file if you don't want to wait 15 minutes
    cgroup = CameraGroup.load('calibration.toml')

    ## example triangulation without filtering, should take < 15 seconds
    pose2d_dir = r'C:\Users\haydenjs\Documents\anipose_demo\hand-demo-unfilled\2019-08-02\pose-2d'
    fname_dict = {
        'A': os.path.join(pose2d_dir, '2019-08-02-vid01-camA.h5'),
        'B': os.path.join(pose2d_dir, '2019-08-02-vid01-camB.h5'),
        'C': os.path.join(pose2d_dir, '2019-08-02-vid01-camC.h5'),
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

    plt.figure(figsize=(9.4, 6))
    plt.plot(p3ds[:, 0, 0])
    plt.plot(p3ds[:, 0, 1])
    plt.plot(p3ds[:, 0, 2])
    plt.xlabel("Time (frames)")
    plt.ylabel("Coordinate (mm)")
    plt.title("x, y, z coordinates of {}".format(bodyparts[0]))

    ## plot the first frame in 3D
    from mpl_toolkits.mplot3d import Axes3D

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
    ax.scatter(p3d[:, 0], p3d[:, 1], p3d[:, 2], c='black', s=100)
    connect_all(ax, p3d, scheme, bodyparts)
