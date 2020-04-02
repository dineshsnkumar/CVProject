import cv2
import numpy as np
import matplotlib.pyplot as plt
from RANSAC import ransac
from StichImages import *
from FeatureDetectorandExtraction import *


def main():
    # # Part 1 FEATURE DETECTION AND MATCHING
    # img1 = cv2.imread('project_images/Rainier1.png')
    # img2 = cv2.imread('project_images/Rainier2.png')
    #
    # img7 = img1.copy()
    # img8 = img2.copy()
    #
    # # Convert to gray scale
    # gray_img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    # gray_img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    #
    #
    # box_img = cv2.imread('project_images/Boxes.png')
    #
    # corners, corner_box_img = harris_corner_feature_detector(box_img)
    # corners1, corner_box_img1 = harris_corner_feature_detector(img7)
    # corners2, corner_box_img2 = harris_corner_feature_detector(img8)
    # cv2.imwrite('./output/1a.png', corner_box_img)
    # cv2.imwrite('./output/1b.png', corner_box_img1)
    # cv2.imwrite('./output/1c.png', corner_box_img2)

    # Part 1 FEATURE DETECTION AND MATCHING
    img1 = cv2.imread('./myoutput/46.png')
    # img2 = cv2.imread('project_images/Rainier3.png')
    img2 = cv2.imread('./myoutput/345.png')


    # cv2.circle(img1,(517,0),6, (0, 255, 0), -1 )
    # cv2.circle(img1, (517, 388), 6, (0, 255, 0), -1)
    # cv2.imshow('img', img1)
    # cv2.waitKey()

    # Convert to gray scale
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Part 3 PANORAMA MOSAIC STITCHING
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_img1, None)
    kp2, des2 = sift.detectAndCompute(gray_img2, None)

    # Match KeyPoints in both the images
    bfmatcher = cv2.BFMatcher()
    matches = bfmatcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img7 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None)
    # cv2.imshow('', img3)
    # cv2.waitKey()

    # Perform Ransac
    hom, invHom, inLierMatchesList = ransac(matches, 4, 800, 10, img1, img2, kp1, kp2)
    inlierImg = cv2.drawMatches(img1, kp1, img2, kp2, inLierMatchesList[:10], None)
    # cv2.imshow('', inlierImg)
    # cv2.waitKey()

    # Merge the images
    stichedImage = stitch(img1, img2, hom, invHom)
    cv2.imwrite("./myoutput/123456_2.png", stichedImage)


    # # Multiple Image Stitching
    '''
    Stitch img1, img2 output stitchedImage with img3
    '''

    img3 = cv2.imread('project_images/Rainier3.png')
    img4 = cv2.imread('project_images/Rainier4.png')
    img5 = cv2.imread('project_images/Rainier5.png')
    img6 = cv2.imread('project_images/Rainier6.png')






if __name__ == '__main__':
    main()
