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
	# need an all black pattern
}

# takes marker id and returns the marker
def id2marker(marker_id):
	for key, val in aruco_dictionary.items():
		if val == marker_id:
			return key
	return None
# adds a white border 3 pixels wide around the marker: 6x6 -> 12x12
def add_border(marker):
	marker_with_border = np.ones((12, 12))
	for i in range(12):
		for j in range(12):
			if 3 <= i and i <= 9 and 3 <= j and j <= 9:
				val1 = marker_with_border[i, j]
				val2 = marker[i - 3, j - 3]
				marker_with_border[i, j] = marker[i - 3, j - 3]
	return marker_with_border
# charuco board: ids 00-39 written as their #, black is -1
charuco = [
		[0, -1, 1, -1, 2, -1, 3, -1, 4, -1],
		[-1, 5, -1, 6, -1, 7, -1, 8, -1, 9],
		[10, -1, 11, -1, 12, -1, 13, -1, 14, -1],
		[-1, 15, -1, 16, -1, 17, -1, 18, -1, 19],
		[20, -1, 21, -1, 22, -1, 23, -1, 24, -1],
		[-1, 25, -1, 26, -1, 27, -1, 28, -1, 29],
		[30, -1, 31, -1, 32, -1, 33, -1, 34, -1],
		[-1, 35, -1, 36, -1, 37, -1, 38, -1, 39]
	] # -1 is black
# black marker (12x12)
black = [
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	]
# makes a board within the 6x6_250 aruco library
def make_board(marker_size):
	image_size = marker_size + 2  # * border_size
	image = np.ones((image_size, image_size), dtype=np.uint8) * 255  # 255 creates a white background
	grid_size = marker_size // 24
	step_size = grid_size // 2
	for i in range(8): # charuco rows
		for j in range(10): # charuco cols
			pattern = charuco[j][i]
			if pattern == '-1':
				pattern = black
			else:
				pattern = id2marker(pattern)
			pattern = add_border(pattern)
			for k in range(12):
				for l in range(12):
					x = (k * grid_size) + step_size
					y = (l * grid_size) + step_size
					if pattern[l][k] == 0:
						cv2.rectangle(image, (x - step_size, y - step_size), (x + step_size, y + step_size), 0, -1)
	cv2.imshow('ChArUco', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# generate tag
aruco_type = "DICT_6X6_250"
id = 1 # helpful if we want to generate multiple of the same pattern

aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

print("ArUCo type '{}' with ID '{}'".format(aruco_type, id))
tag_size = 250
tag = np.zeros((tag_size, tag_size, 1), dtype="uint8")
cv2.aruco.drawMarker(aruco_dictionary, id, tag_size, tag, 1)

# save tag
tag_name = "arucoMarkers/" + aruco_type + "_" + str(id) + ".png"
cv2.imwrite(tag_name, tag)
cv2.imshow("ArUCo Tag", tag)

cv2.waitKey(0)

cv2.destroyAllWindows()
