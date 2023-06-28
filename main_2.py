import numpy as np
from aniposelib.boards import CharucoBoard, Checkerboard
from aniposelib.cameras import Camera, CameraGroup
from aniposelib.utils import load_pose2d_fnames

import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
import os
from datetime import datetime

import crop_videos
import skilled_reaching_io

def connect(ax, points, bps, bp_dict, color):
    ixs = [bp_dict[bp] for bp in bps]
    return ax.plot(points[ixs, 0], -points[ixs, 1], points[ixs, 2], color=color)

def connect_all(ax, points, scheme, bodyparts, cmap=None):
    if cmap is None:
        cmap = get_cmap('tab10')
    bp_dict = dict(zip(bodyparts, range(len(bodyparts))))
    lines = []
    for i, bps in enumerate(scheme):
        line = connect(ax, points, bps, bp_dict, color=cmap(i)[:3])
        lines.append(line)
    return lines

# def calibrate():

if __name__ == '__main__':

    video_root_folder = r'C:\Users\haydenjs\Desktop'
    crop_params_csv_path = os.path.join(video_root_folder, 'SR_video_crop_regions.csv')
    crop_params_df = skilled_reaching_io.read_crop_params_csv(crop_params_csv_path)
    session_date_string = '20230608'  # 06/08/2023
    session_date = datetime.strptime(session_date_string, '%Y%m%d')
    box_num = 1

    view_list = ['direct', 'leftmirror', 'rightmirror']
    crop_params_dict = {
        view_list[0]: [700, 1350, 270, 935],
        view_list[1]: [1, 470, 270, 920],
        view_list[2]: [1570, 2040, 270, 920]
    }

    crop_params_dict = crop_videos.crop_params_dict_from_df(crop_params_df, session_date, box_num, view_list)

    # crop_videos.crop_folders(video_folder_list, cropped_vids_parent, crop_params_dict, view_list, vidtype='avi', filtertype='mjpeg2jpeg')

    dest_folder = r'C:\Users\haydenjs\Desktop\cropped_calibration_samples_for_Hayden'

    vid_path_in = r'C:\Users\haydenjs\Desktop\calibration_samples_for_Hayden'

    vids = [vid for vid in os.listdir(r'C:\Users\haydenjs\Desktop\calibration_samples_for_Hayden') if
            os.path.isfile(os.path.join(r'C:\Users\haydenjs\Desktop\calibration_samples_for_Hayden', vid))]
    vidnames = []
    for vid in vids:
        vidnames.append(vid)

    for i in range(len(vids)):
        vid_path_in = r'C:\Users\haydenjs\Desktop\calibration_samples_for_Hayden\''
        vid_path_in = vid_path_in[:-1] + vidnames[i]
        full_vid_path = vid_path_in
        for j in range(3):
            view_name = view_list[j]
            crop_params = crop_params_dict[view_name]
            vid_path_out = crop_videos.cropped_vid_name(full_vid_path, dest_folder, view_name, crop_params)
            crop_videos.crop_video(vid_path_in, vid_path_out, crop_params, view_name, filtertype='h264')

    cropped_vids = [cropped_vid for cropped_vid in
                    os.listdir(r'C:\Users\haydenjs\Desktop\cropped_calibration_samples_for_Hayden') if
                    os.path.isfile(
                        os.path.join(r'C:\Users\haydenjs\Desktop\cropped_calibration_samples_for_Hayden', cropped_vid))]
    cropped_vidnames = []
    leftmirror = 'leftmirror'
    rightmirror = 'rightmirror'
    flipped = 'flipped'
    for cropped_vid in cropped_vids:
        if leftmirror in cropped_vid and flipped not in cropped_vid:
            continue
        elif rightmirror in cropped_vid and flipped not in cropped_vid:
            continue
        cropped_vid = [cropped_vid]
        cropped_vidnames.append(cropped_vid)
    cropped_vid_path_in = r'C:\Users\haydenjs\Desktop\cropped_calibration_samples_for_Hayden'
    cropped_vidnames = [[os.path.join(cropped_vid_path_in, vn[0])] for vn in cropped_vidnames]

    cam_names = ['A', 'B', 'C']

    n_cams = len(vidnames)

    # 7x10 board: (7, 10, square_length=7.5, marker_length=5.625, marker_bits=4, dict_size=50)
    # 4x5 board: (4, 5, square_length=10, marker_length=7.5, marker_bits=4, dict_size=50)

    board = CharucoBoard(4, 5,
                         square_length=7.5,  # here, in mm but any unit works
                         marker_length=5.625,
                         marker_bits=4, dict_size=50)

    # the videos provided are fisheye, so we need the fisheye option # ours are not fisheye
    cgroup = CameraGroup.from_names(cam_names, fisheye=True)

    # this will take about 15 minutes (mostly due to detection)
    # it will detect the charuco board in the videos,
    # then calibrate the cameras based on the detections, using iterative bundle adjustment
    # cgroup.calibrate_videos(vidnames, board)
    cgroup.calibrate_videos(cropped_vidnames, board)

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

    # my front videos
    # vid_path = 'C:\Users\haydenjs\Desktop\calibration_samples_for_Hayden'
    # 'hayden_front_1.avi', 220/400
    # 'hayden_front_1_left.mov', 0
    # 'hayden_front_1_center.mov', 10
    # 'hayden_front_1_right.mov', 0

    # 'hayden_front_2.avi', 288/400
    # 'hayden_front_2_left.mov', 0
    # 'hayden_front_2_center.mov', 77
    # 'hayden_front_2_right.mov', 0

    # 'hayden_front_3.avi', 0/400
