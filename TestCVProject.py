import cv2
import numpy as np
import matplotlib.pyplot as plt


source_window = 'Source image'
corners_window = 'Corners detected'
max_thresh = 255

def cornerHarris_demo(val, src_gray):
    thresh = val
    # Detector parameters
    blockSize = 2
    apertureSize = 3
    k = 0.04
    # Detecting corners
    dst = cv2.cornerHarris(src_gray, blockSize, apertureSize, k)
    # Normalizing
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    dst_norm_scaled = cv2.convertScaleAbs(dst_norm)
    # Drawing a circle around corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i,j]) > thresh:
                cv2.circle(dst_norm_scaled, (j,i), 5, (0), 2)
    # Showing the result
    cv2.namedWindow(corners_window)
    cv2.imshow(corners_window, dst_norm_scaled)
    cv2.waitKey()


def main():
    src = cv2.imread('project_images/Boxes.png')
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)


    # Create a window and a trackbar
    cv2.namedWindow(source_window)
    thresh = 200  # initial threshold

    cv2.createTrackbar('Threshold: ', source_window, thresh, max_thresh, cornerHarris_demo)
    cv2.imshow(source_window, src)
    cornerHarris_demo(thresh, src_gray)




if __name__ == '__main__':
    main()