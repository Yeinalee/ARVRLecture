import numpy as np
from matplotlib import pyplot as plt

def practice_main():
    imgh, imgw = 250, 250
    n_samples = 2000
    img = np.zeros((imgh, imgw), dtype=np.uint8)

    alpha = np.random.rand(n_samples)
    beta = np.random.rand(n_samples)

    P = np.array([50, 50])
    PQ = np.array([150, 50])
    PR = np.array([100, 100])

    #tri = P[:, np.newaxis] + alpha*(1-beta)*PQ[:,np.newaxis] + beta*PR[:, np.newaxis]
    v = alpha + beta <= 1
    tri= P[:, np.newaxis] + alpha[v]*PQ[:, np.newaxis] + beta[v]*PR[:, np.newaxis]


    x, y =tri.astype(np.int32)

    img[y, x] = 255
    plt.imshow(img, cmap='gray')
    plt.show()

if __name__ == '__main__':
    practice_main()