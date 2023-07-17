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
