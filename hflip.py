    if view_name == 'leftmirror' or 'rightmirror':
        # flip the video we just wrote out horizontally, give it a new name with "_flip" at the end
        # flipped_vid_name = (vid_name) + _flip
        hflip_command = (
            f"ffmpeg - i {vid_path_out} "
            f"-vf hflip "
            f"-c:a {vid_path_out}"
        )
        subprocess.call(hflip_command, shell=True)
