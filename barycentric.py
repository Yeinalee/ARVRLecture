import numpy as np
from matplotlib import pyplot as plt

def practice_main():
    imgh, imgw = 250, 300
    img = np.zeros((imgh, imgw, 3), dtype = np.uint8)

    P = np.array([50, 50])
    Q = np.array([200, 100])
    R = np.array([150, 150])
    PQ = Q - P
    PR = R -P

    # 범위 지
    tri_max = np.max(np.array([P, Q, R]), axis=0) + 1
    tri_min = np.min(np.array([P, Q, R]), axis=0)
    X = np.mgrid[tri_min[0]: tri_max[0], tri_min[1]:tri_max[1]]

    PX = X - P[:, np.newaxis, np.newaxis]
    A = np.cross(PQ, PR)

    l1 = np.cross(PX, PR[:, np.newaxis, np.newaxis], axis = 0) / A
    l2 = np.cross(PQ[:, np.newaxis, np.newaxis], PX, axis = 0) / A
    v = (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
    x, y = X[0,v], X[1, v]

    P_Color = np.array([255, 0, 0])
    Q_Color = np.array([0, 255, 0])
    R_Color = np.array([0, 0, 255])

    C = l1[v] * Q_Color[:, np.newaxis] + l2[v] * R_Color[:, np.newaxis] + (1 - l1[v] - l2[v]) * P_Color[:, np.newaxis]
    img[y, x,:] = C.T
    plt.imshow(img, cmap='gray')
    plt.show()

if __name__ == '__main__':
    practice_main()