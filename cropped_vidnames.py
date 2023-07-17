video_root_folder = r'C:\Users\haydenjs\Desktop'
    crop_params_csv_path = os.path.join(video_root_folder, 'SR_video_crop_regions.csv')
    crop_params_df = skilled_reaching_io.read_crop_params_csv(crop_params_csv_path)
    session_date_string = '20230608'  # 06/08/2023
    session_date = datetime.strptime(session_date_string, '%Y%m%d')
    box_num = 1

    view_list = ['direct', 'leftmirror', 'rightmirror']

    crop_params_dict = {
        view_list[0]: [600, 1450, 1, 935],
        view_list[1]: [1, 500, 1, 935],
        view_list[2]: [1600, 2040, 1, 935]
    } # left, right, top, bottom

    # crop_params_dict = crop_videos.crop_params_dict_from_df(crop_params_df, session_date, box_num, view_list)

    # crop_videos.crop_folders(video_folder_list, cropped_vids_parent, crop_params_dict, view_list, vidtype='avi', filtertype='mjpeg2jpeg')

    # anipose vidnames
    # vidnames = [['calib-charuco-camA.MOV'],
    # ['calib-charuco-camB.MOV'],
    # ['calib-charuco-camC.MOV']]

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

# to put at bottom of Dan's crop_videos.py

    if view_name != 'direct':
        # flip the video we just wrote out horizontally, give it a new name with "_flipped" at the end
        # temporarily removes the video's file extension
        flipped_vid_path_out = vid_path_out.rsplit(".", 1)[0] + '_flipped.avi'
        hflip_command = (
            f"ffmpeg -i {vid_path_out} "
            f"-vf hflip "
            f"-c:a copy {flipped_vid_path_out}"
        )
        subprocess.call(hflip_command, shell=True)
