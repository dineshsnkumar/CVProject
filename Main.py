import cv2
import numpy as np
import matplotlib.pyplot as plt
from RANSAC import ransac
from StichImages import *
from FeatureDetectorandExtraction import *
from SIFTDescrptor import *


def main():

    # Part 1 -----------FEATURE DETECTION ------------------------------------
    img1 = cv2.imread('project_images/Rainier1.png')
    img2 = cv2.imread('project_images/Rainier2.png')
    iterations = 1000

    img7 = img1.copy()
    img8 = img2.copy()

    # Convert to gray scale
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    box_img = cv2.imread('project_images/Boxes.png')

    corners, corner_box_img = harris_corner_feature_detector(box_img)
    corners1, corner_box_img1 = harris_corner_feature_detector(img7)
    corners2, corner_box_img2 = harris_corner_feature_detector(img8)

    cv2.imwrite('./output/1a.png', corner_box_img)
    cv2.imwrite('./output/1b.png', corner_box_img1)
    cv2.imwrite('./output/1c.png', corner_box_img2)

    #  Part 3 ------------PANORAMA MOSAIC STITCHING - Two Images-----------------------
    print('---Panorama Stitching started -----')
    sift = cv2.xfeatures2d.SIFT_create()
    matches, kp1, kp2 = siftMatches(sift, gray_img1, gray_img2)

    # Perform Ransac
    hom, invHom, inLierMatchesList = ransac(matches, 4, iterations, 10, img1, img2, kp1, kp2)
    inlierImg = cv2.drawMatches(img1, kp1, img2, kp2, inLierMatchesList[:10], None)
    cv2.imwrite("./output/3.png", inlierImg)

    # Merge the images
    stichedImage = stitch(img1, img2, hom, invHom)
    print('Stitched Images 1 and 2 at ./output/4.png')
    cv2.imwrite("./output/4.png", stichedImage)





if __name__ == '__main__':
    main()
