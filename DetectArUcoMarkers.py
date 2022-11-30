import cv2
from cv2 import aruco
import numpy as np

marker_length = .06 #교수님은 0.06
marker_spacing = .01
cols = 4
rows = 4

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
aruco_params = aruco.DetectorParameters_create()
board = aruco.GridBoard_create(cols, rows, marker_length, marker_spacing, aruco_dict)

img = cv2.imread("board1.jpg")

corners, ids, _ = aruco.detectMarkers(img, aruco_dict, parameters = aruco_params)
# detected_img = aruco.drawDetectedMarkers(img.copy(), corners, ids)

# cv2.imshow('Original', img)
# cv2.imshow('Detected', detected_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

_, mtx, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraAruco(corners, ids, np.array(len(ids)), board, img.shape[:2], None, None)
aruco.calibrateCamersAruco(corners, ids, np.array(len(ids), board, img.shape[:2], None, None)

rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, mtx, dist_coeffs)

#img_aruco = cv2.drawFrameAxes(img, mtx, dist_coeffs, rvecs[0], tvecs[0], marker_length)

for i in range(len(rvec)):
    img_aruco = cv2.drawFrameAxes(img, mtx, dist_coeffs, rvec[i], tvec[i], marker_length)

cv2.imshow('Detected', img_aruco)
cv2.waitKey(0)