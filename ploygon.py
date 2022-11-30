import numpy as np
import cv2
from matplotlib import pyplot as plt

def practice_main():
    img = np.zeros((500, 500, 3), dtype = np.uint8)
    points1 = np.array([[100, 50], [120, 180], [50, 250]])
    points2 = np.array([[270, 130], [220, 50], [290, 50]])
    points3 = points1 * 2
    points4 = points2 + np.array([100, 50])

    theta = np.radians(30)
    rotm = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    points5 = (points3 @ rotm.T).astype(np.int32) #matrix 곱
    points6 = (points4 @ rotm.T).astype(np.int32)


    cv2.fillPoly(img, [points1, points2], color=(0, 255, 0))
    cv2.fillPoly(img, [points3, points4], color=(255, 0, 0))
    cv2.fillPoly(img, [points5, points6], color = (0, 0, 255))

    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    #practice_main()

    x = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
    print(x[np.array([10,9,8,8,8])]) # [10  9  8  8  8]
    print(x[[10,9,8,8,8]]) # np.array() 생략 가능

    x = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
    print(x[1:3, 1:3]) # slicing [[4 5]
                                # [7 8]]
    print(x[np.arange(1,3), np.arange(1,3)]) # [4 8]
    print(x[1:3, np.arange(1,3)]) #[[4 5]
                                  # [7 8]]
    print(x[1:3, [2,1]]) #[[5 4]
                         # [8 7]]
    print(x[[1,2], [2,1]]) #[5 7]
