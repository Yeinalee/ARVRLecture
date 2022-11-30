import cv2
from cv2 import aruco

marker_length = .05 #교수님은 0.06
marker_spacing = .01
cols = 5 #4
rows = 5 #4

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
board = aruco.GridBoard_create(cols, rows, marker_length, marker_spacing, aruco_dict)
img = cv2.aruco_GridBoard.draw(board, (1000, 1000))


cv2.imwrite("board.png", img)
cv2.imshow("aruco", img)
cv2.waitKey()


