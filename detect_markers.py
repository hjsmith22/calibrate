def detect_markers(image):
	# load dictionary
	aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

	# convert the image to grayscale
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	_, threshold = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
	contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	corners = []
	ids = []

	for contour in contours:
		epsilon = 0.02 * cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour, epsilon, True)

		if len(approx) == 4:  # Only consider contours with 4 corners
			# Calculate the area and aspect ratio of the contour
			area = cv2.contourArea(contour)
			x, y, w, h = cv2.boundingRect(approx)
			aspect_ratio = w / float(h)

			# Detect if the contour meets the ChArUco marker criteria
			if 500 < area < 1000 and 0.7 < aspect_ratio < 1.3:
				corners.append(approx.reshape(-1, 2))
				ids.append(-1)  # Assign a placeholder ID for each marker

	new_image = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)

	cv2.imshow('ArUco Markers', new_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return new_image, corners, ids

if __name__ == '__main__':
	# Load the input image
	# marker_type = cv2.aruco.DICT_6X6_250
	image = cv2.imread(r"C:\Users\haydenjs\Desktop\screenshot.png")
	new_image, corners, ids = detect_markers(image)
