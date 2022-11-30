import numpy as np
from matplotlib import pyplot as plt

def rot_matrix(x_degree, y_degree, z_degree): #회전 함수
    x_rad = np.radians(x_degree)
    y_rad = np.radians(y_degree)
    z_rad = np.radians(z_degree)

    rotmX = np.array([[1, 0, 0], [0, np.cos(x_rad), -np.sin(x_rad)], [0, np.sin(x_rad), np.cos(x_rad)]])
    rotmY = np.array([[np.cos(y_rad), 0, np.sin(y_rad)], [0, 1, 0], [-np.sin(y_rad), 0, np.cos(y_rad)]])
    rotmZ = np.array([[np.cos(z_rad), -np.sin(z_rad), 0], [np.sin(z_rad), np.cos(z_rad), 0], [0, 0, 1]])

    return rotmZ @ rotmY @ rotmX


def read_obj(obj_file):
    v = []
    f = []
    c = []
    with open(obj_file, "r") as fp:
        for s in fp.readlines(): #각라인, 개행문자 포함
            cols = s[:-1].split() #한줄씩 읽어서 공백을 기준으로 나눠서 리스트로, 개행문자 제외
            if len(cols) > 0:
                what = cols[0].lower() #제일 앞 문자 소문자로 바꿔주기, v / f 확인

                if what == 'v': #vertex일때
                    assert(len(cols) == 4 or len(cols) == 7) #조건을 만족하지 않으면 종료
                    v.append(cols[1:4])
                    if len(cols) == 7: #색상 포함
                        c.append(cols[4:7])

                elif what == 'f':
                    assert (len(cols) == 4)
                    f.append(cols[1:])

    return np.array(v, dtype = np.float32), np.array(f, dtype = int), np.array(c, dtype = np.float32)

def practice_main(*rot):
    v, f, c = read_obj('Object1.obj')
    print('Before', v.min(axis=0), v.max(axis = 0))
    obj_bound = v.max(axis = 0) - v.min(axis = 0)
    print('Object Bound : ', obj_bound)
    obj_center = v.min(axis = 0) + obj_bound/2
    print('Object Center : ', obj_center)

    imgh, imgw = 300, 300
    scale = 10
    trans = (imgw/2, imgh/2, obj_center[2]) - obj_center
    v = (v - obj_center) @ rot_matrix(*rot).T * scale + obj_center + trans
    # 중앙으로 옮긴 후 scale -> 다시 옮기기

    print('After', v.min(axis = 0), v.max(axis = 0))
    obj_bound = v.max(axis = 0) - v.min(axis = 0)
    print('Object Bound : ', obj_bound)
    obj_center = v.min(axis=0) + obj_bound / 2
    print('Object Center : ', obj_center)

    img = np.zeros((imgh, imgw, 3), dtype = np.uint8) #도화지 만들기

    # for face in f: #각 면에 대해서 (1 X 3)
    #     print(face)

    for face in f:
        tri_verts = v[face-1, :] #파이썬 배열이 0부터 시작

        tri_max = np.floor(np.max(tri_verts, axis = 0)).astype(int) + 1
        tri_min = np.ceil(np.min(tri_verts, axis=0)).astype(int)

        X = np.mgrid[tri_min[0]: tri_max[0], tri_min[1]:tri_max[1]]
        P = tri_verts[0, :2]
        Q = tri_verts[1, :2]
        R = tri_verts[2, :2]
        PQ = Q - P
        PR = R - P
        PX = X - P[:, np.newaxis, np.newaxis]
        A = np.cross(PQ, PR)

        l1 = np.cross(PX, PR[:, np.newaxis, np.newaxis], axis=0) / A
        l2 = np.cross(PQ[:, np.newaxis, np.newaxis], PX, axis=0) / A

        val = (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)

        if np.any(val):
            x, y = X[0, val], X[1, val]
            tri_colors = c[face-1, :]
            P_Color = tri_colors[0]
            Q_Color = tri_colors[1]
            R_Color = tri_colors[2]

            C = l1[val] * Q_Color[:, np.newaxis] + l2[val]*R_Color[:, np.newaxis] + (1-l1[val]-l2[val])*P_Color[:,np.newaxis]
            img[y, x, :] = C.T * 255

    plt.imshow(img)
    plt.show()

def practice_main2(*rot): #z-buffer 추가
    v, f, c = read_obj('Object1.obj')
    obj_bound = v.max(axis = 0) - v.min(axis = 0)
    obj_center = v.min(axis = 0) + obj_bound/2

    imgh, imgw = 300, 300
    scale = 10
    trans = (imgw/2, imgh/2, obj_center[2]) - obj_center
    v = (v - obj_center)  @ rot_matrix(*rot).T * scale + obj_center + trans
    # 중앙으로 옮긴 후 scale -> 다시 옮기기

    img = np.zeros((imgh, imgw, 3), dtype = np.uint8) #도화지 만들기
    zbuffer = np.ones((imgh, imgw))*np.inf #무한대로 초기화 -> 최대한 큰 수로!

    for face in f:
        tri_verts = v[face-1, :] #파이썬 배열이 0부터 시작

        tri_max = np.floor(np.max(tri_verts, axis = 0)).astype(int) + 1
        tri_min = np.ceil(np.min(tri_verts, axis=0)).astype(int)

        X = np.mgrid[tri_min[0]: tri_max[0], tri_min[1]:tri_max[1]]
        P = tri_verts[0, :2]
        Q = tri_verts[1, :2]
        R = tri_verts[2, :2]
        PQ = Q - P
        PR = R - P
        PX = X - P[:, np.newaxis, np.newaxis]
        A = np.cross(PQ, PR)

        l1 = np.cross(PX, PR[:, np.newaxis, np.newaxis], axis=0) / A
        l2 = np.cross(PQ[:, np.newaxis, np.newaxis], PX, axis=0) / A

        val = (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)

        if np.any(val):
            x, y = X[0, val], X[1, val]

            P_Z = tri_verts[0, 2]
            Q_Z = tri_verts[1, 2]
            R_Z = tri_verts[2, 2]
            Z = l1[val] * Q_Z + l2[val]*R_Z + (1 - l1[val] - l2[val]) * P_Z

            testZ = Z < zbuffer[y, x]
            tri_colors = c[face-1, :]

            if np.any(testZ):
                zbuffer[y[testZ], x[testZ]] = Z[testZ]
                tri_colors = c[face-1, :]
                P_Color = tri_colors[0]
                Q_Color = tri_colors[1]
                R_Color = tri_colors[2]

                C = l1[val] * Q_Color[:, np.newaxis] + l2[val]*R_Color[:, np.newaxis] + (1-l1[val]-l2[val])*P_Color[:,np.newaxis]
                img[y, x, :] = C.T * 255

    plt.imshow(img)
    plt.show()

    #깊이 관계 확인
    zbuffer = zbuffer - np.min(zbuffer)
    zbuffer[np.isinf(zbuffer)] = 0
    zbuffer = zbuffer / np.max(zbuffer)

    plt.imshow(zbuffer, cmap='gray')
    plt.show()



if __name__ == '__main__':
    practice_main(90, 0, 0)
    practice_main2(90, 0, 0)