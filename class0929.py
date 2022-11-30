import numpy as np
import cv2
from matplotlib import pyplot as plt

def practice_main():
    img = np.zeros([300, 300, 3], dtype = np.uint8)
    img[:50, :, 0] = 255
    img[50:200, :, 1] = 255
    img[200:300, :, 2] = 255

    plt.imshow(img)
    plt.show()

    cv2.imshow('Picture', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    practice_main()


