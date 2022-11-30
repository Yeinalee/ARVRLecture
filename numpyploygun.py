import numpy as np
from matplotlib import pyplot as plt

def practice_main():
    imgh, imgw = 100, 150
    img = np.zeros((imgh, imgw), dtype=np.uint8)
    y, x = np.mgrid[0:imgh, 0:imgw]

    img[y, x] = ((x-50)**2 + (y-50)**2 - 40**2 <= 0) * 255
    #모든 픽셀이 조건을 만족하는지 확인

    plt.imshow(img, cmap='gray')
    plt.show()

    y, x = np.mgrid[10:91, 10:91]

    img[y, x] = ((x-50)**2 + (y-50)**2 - 40**2 <= 0) *255
    plt.imshow(img, cmap='gray')
    plt.show()

if __name__ == '__main__':
    practice_main()