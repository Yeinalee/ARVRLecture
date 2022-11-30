import numpy as np
import cv2
from matplotlib import pyplot as plt

def practice_main():
    imgh, imgw = 100, 100
    n_of_samples = 2000
    img = np.zeros((imgh, imgw), dtype=np.uint8)

    y = np.random.randint(imgh, size=n_of_samples)
    x = np.random.randint(imgw, size=n_of_samples)

    #img[y, x] = 255 # 임의의 x,y
    v = (x-50)**2 + (y-50)**2 - 40**2 <= 0 #True, False결과 값 n_of_samples 개
    img[y[v], x[v]] = 255

    plt.imshow(img, cmap='gray')
    plt.show()

if __name__ == '__main__':
    practice_main()