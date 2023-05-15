import numpy as np
import cv2.aruco

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

'''
array(['../../data/calib_tel_ludo/VID_20180406_085421_0.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_5.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_10.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_15.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_20.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_25.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_30.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_35.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_40.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_45.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_50.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_55.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_60.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_65.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_70.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_75.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_80.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_85.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_90.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_95.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_100.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_105.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_110.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_115.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_120.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_125.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_130.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_135.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_140.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_145.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_150.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_155.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_160.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_165.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_170.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_175.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_180.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_185.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_190.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_195.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_200.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_205.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_210.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_215.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_220.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_225.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_230.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_235.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_240.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_245.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_250.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_255.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_260.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_265.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_270.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_275.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_280.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_285.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_290.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_295.png',
           '../../data/calib_tel_ludo/VID_20180406_085421_300.png'],
          dtype='<U53')
'''