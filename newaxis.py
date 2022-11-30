import numpy as np
from matplotlib import pyplot as plt

def practice_main():
    a = np.array([0.0, 10.0, 20.0, 30.0])
    b = np.array([1.0, 2.0, 3.0])


    print(np.reshape(a, (-1,1)) + b) #a를 눕혀버림
    print(a[:, np.newaxis] + b)
if __name__ == '__main__':
    practice_main()