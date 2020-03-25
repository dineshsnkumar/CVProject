import cv2
import numpy as np
import matplotlib.pyplot as plt



def main():
    point1 = [[1, 2]]
    point2 = [[3, 4]]
    points3 = [[5,6]]

    points = np.array([point1, point2, points3], np.float32)
    print(points)
    print(cv2.boundingRect(points))





if __name__ == '__main__':
    main()