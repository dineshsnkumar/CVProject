import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from RANSAC import *

'''
Stitch the images using computed Homography
@:argument: img1, img2, hom, inv_hom 
@:return StitchedImage
'''

def stitch(img1, img2, H, inv_H):
    print('----Stitching Images ------- ')
    # Find StitchedImage size
    # print('Image1 dimensions', img1.shape)

    # Four Points of Image1
    h1, w1, d1 = img1.shape
    img1points1 = [[0, 0]]
    img1points2 = [[w1, 0]]
    img1points3 = [[0, h1]]
    img1points4 = [[w1, h1]]

    # points1 = np.array([img1points1, img1points2, img1points3, img1points4], np.float32)

    # print('Points1 Image', points1)

    # Four points of Image2
    h2, w2, d2 = img2.shape
    x, y = project(0, 0, inv_H)
    x1, y1 = project(w2, 0, inv_H)
    x2, y2 = project(0, h2, inv_H)
    x3, y3 = project(w2, h2, inv_H)

    img2points1 = [[x, y]]
    img2points2 = [[x1, y1]]
    img2points3 = [[x2, y2]]
    img2points4 = [[x3, y3]]

    points = np.array(
        [img1points1, img1points2, img1points3, img1points4, img2points1, img2points2, img2points3, img2points4],
        np.float32)

    boundaryPoints = cv2.boundingRect(points)
    w = boundaryPoints[2] - boundaryPoints[0]
    h = boundaryPoints[3] - boundaryPoints[1]
    stichedImage = np.zeros([h, w, 3], np.uint8)

    # cv2.imshow('Stitched Window', stichedImage)
    # cv2.waitKey()

    print('Image 1 dimensions', img1.shape, 'Image2 dimensions', img2.shape)
    print('Image 1 boundary points', img1points1, img1points2, img1points3, img1points4)
    print('Image 2 boundary', (0, 0), (w2, 0), (0, h2), (w2, h2))
    print('Projected Point', img2points1, img2points2, img2points3, img2points4)
    print('Boundary Points', boundaryPoints)
    print('Stitched Image', stichedImage.shape)

    # Copy Image 1 to stichedImage
    # stichedImage[0:img1.shape[0], 0:img1.shape[1]] = img1
    for y in range(0, img1.shape[0]):
        for x in range(0, img1.shape[1]):
            stichedImage[y - boundaryPoints[1], x - boundaryPoints[0]] = img1[y,x]



    # Project the stitchedImage in image2 space
    '''
    1. Project each pixel in Stitched Image to image2 
    2. IF the point lies within image2 boundaries add pixel to stitchedImage 
    '''

    for y in range(boundaryPoints[1], stichedImage.shape[0]):
        for x in range(boundaryPoints[0], stichedImage.shape[1]):
            x1, y1 = project(x,y, H[0])
            # print('Projected Points', x1, y1)
            if(0 <= x1 < img2.shape[1]) and (0 <= y1 < img2.shape[0]):
                pixelValueImg2 = cv2.getRectSubPix(img2,(1,1), (x1, y1) )
                if (y-boundaryPoints[1] < stichedImage.shape[0]) and (x-boundaryPoints[0]< stichedImage.shape[1]):
                    stichedImage[y - boundaryPoints[1], x - boundaryPoints[0]] = pixelValueImg2[0][0]


    print('-------Exiting the Stitched Image -----------')
    return stichedImage


def stitchMultipleImages(stichedImage, img, sift):
    print('----------Stitching Multiple Images together')

    stichedImage_gray = cv2.cvtColor(stichedImage, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(stichedImage_gray, None)
    kp2, des2 = sift.detectAndCompute(gray_img, None)

    # Match KeyPoints in both the images
    bfmatcher = cv2.BFMatcher()
    matches = bfmatcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    hom, invHom, inLierMatchesList = ransac(matches, 4, 400, 10, stichedImage, img, kp1, kp2)
    stichedImage2 = stitch(stichedImage, img, hom, invHom)
    cv2.imshow('', stichedImage2)
    cv2.waitKey()
    return stichedImage2
