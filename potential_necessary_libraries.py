import numpy as np
import cv2
import os
from aniposelib.boards import CharucoBoard, Checkerboard
from aniposelib.cameras import Camera, CameraGroup
from aniposelib.utils import load_pose2d_fnames
import matplotlib.pyplot as plt
# % matplotlib notebook
from matplotlib.pyplot import get_cmap

import glob
import os
# from moviepy.editor import *
import subprocess
import shutil
import pandas as pd
from datetime import datetime
from tqdm import tqdm

import skilled_reaching_calibration
import navigation_utilities
import crop_videos
import skilled_reaching_io

from typing import Mapping
from typing import cast
from collections.abc import Mapping as ABCMapping
