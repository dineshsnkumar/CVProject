import cv2
import numpy as np
import matplotlib.pyplot as plt
from RANSAC import ransac
from StichImages import *
from FeatureDetectorandExtraction import *


def main():
    # Feature Detection
    img1 = cv2.imread('project_images/Rainier1.png')
    img2 = cv2.imread('project_images/Rainier2.png')

    # img1 = cv2.imread('project_images/MelakwaLake1.png')
    # img2 = cv2.imread('project_images/Me lakwaLake2.png')

    # # # Feature Detection
    # img1 = cv2.imread('project_images/Rainier6.png')
    # img2 = cv2.imread('12345.png')

    # Convert to gray scale
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    box_img = cv2.imread('project_images/Boxes.png')
    # corners, corner_box_img = harris_corner_feature_detector(box_img)
    # cv2.imwrite('1a.png', corner_box_img)

    # Feature Matching
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_img1, None)
    kp2, des2 = sift.detectAndCompute(gray_img2, None)

    # Match KeyPoints in both the images
    bfmatcher = cv2.BFMatcher()
    matches = bfmatcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None)
    # cv2.imshow('', img3)
    # cv2.waitKey()
    cv2.imwrite("2.png", img3)

    # Perform Ransac
    hom, invHom, inLierMatchesList = ransac(matches, 4, 700, 10, img1, img2, kp1, kp2)
    inlierImg = cv2.drawMatches(img1, kp1, img2, kp2, inLierMatchesList[:10], None)
    # cv2.imwrite("3.png", inlierImg)
    # cv2.imshow('', inlierImg)
    # cv2.waitKey()

    # Merge the images
    stichedImage = stitch(img1, img2, hom, invHom)
    cv2.imwrite("4.png", stichedImage)
    cv2.imshow('', stichedImage)
    cv2.waitKey()

    # # Multiple Image Stitching
    '''
    Stitch img1, img2 output stitchedImage with img3
    '''
    # img3 = cv2.imread('project_images/Rainier3.png')
    # stiched3Image = stitchMultipleImages(stichedImage, img3, sift)
    # cv2.imwrite("stiched3Image.png", stiched3Image)

    # img4 = cv2.imread('project_images/Rainier4.png')
    # stiched4Image = stitchMultipleImages(stiched3Image, img4, sift)
    # cv2.imshow('4 Images', stiched4Image)
    # cv2.waitKey()




if __name__ == '__main__':
    main()
