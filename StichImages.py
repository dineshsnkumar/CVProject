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
    x1, y1 = project(0, 0, inv_H)
    x2, y2 = project(w2, 0, inv_H)
    x3, y3 = project(0, h2, inv_H)
    x4, y4 = project(w2, h2, inv_H)

    img2points1 = [[y1, x1]]
    img2points2 = [[y2, x2]]
    img2points3 = [[y3, x3]]
    img2points4 = [[y4, x4]]

    points = np.array(
        [img1points1, img1points2, img1points3, img1points4, img2points1, img2points2, img2points3, img2points4],
        np.float32)
    min__points = np.amin(points, axis=0)
    max_points = np.amax(points, axis=0)
    print('Points', points, 'Min and Max along axis', min__points, max_points)

    boundaryPoints = cv2.boundingRect(points)
    h = boundaryPoints[2] - boundaryPoints[0]
    w = boundaryPoints[3] - boundaryPoints[1]
    stichedImage = np.zeros([h, w, 3], np.uint8)

    print('Image 1 dimensions', img1.shape, 'Image2 dimensions', img2.shape)
    print('Image 1 boundary points', img1points1, img1points2, img1points3, img1points4)
    print('Image 2 boundary', (0, 0), (w2, 0), (0, h2), (w2, h2))
    print('Projected Point', img2points1, img2points2, img2points3, img2points4)
    print('Boundary Points', boundaryPoints)
    print('Stitched Image', stichedImage.shape)

    # Copy Image 1 to stichedImage
    # stichedImage[0:img1.shape[0], 0:img1.shape[1]] = img1
    for y1 in range(0, img1.shape[0]):
        for x1 in range(0, img1.shape[1]):
            stichedImage[y1 - boundaryPoints[0], x1 - boundaryPoints[1]] = img1[y1, x1]


    # Project the stitchedImage in image2 space
    '''
    1. Project each pixel in Stitched Image to image2 
    2. IF the point lies within image2 boundaries add pixel to stitchedImage 
    '''

    for y1 in range(boundaryPoints[0], stichedImage.shape[0]):
        for x1 in range(boundaryPoints[1], stichedImage.shape[1]):
            # The projected points are x, y
            x2, y2 = project(x1, y1, H[0])
            print('x2, y2 is ', x2, y2)
            if (x2 >= 0 and y2 >= 0) and (x2 < img2.shape[1] and y2 < img2.shape[0]):
                pixelValueImg2 = cv2.getRectSubPix(img2, (1, 1), (x2, y2))
                stichedImage[y1 - boundaryPoints[0], x1 - boundaryPoints[1]] = pixelValueImg2[0][0]
                # if y1 < img1.shape[0] :
                print('y1 is ', y1, 'Condition is ', img1.shape[0] - boundaryPoints[0])
                print('y1-boundary[0] ', y1 - boundaryPoints[0])
                    # stichedImage[y1 - boundaryPoints[0], x1 - boundaryPoints[1]] = pixelValueImg2[0][0]

    # for y1 in range(boundaryPoints[0], stichedImage.shape[0]):
    #     for x1 in range(boundaryPoints[1], stichedImage.shape[1]):
    #         x2, y2 = project(x1, y1, H[0])
    #         if (x2 >= 0 and y2 >= 0) and (
    #                 (x2 - boundaryPoints[2]) < img2.shape[0] and (y2 - boundaryPoints[3]) < img2.shape[1]):
    #             pixelValueImg2 = cv2.getRectSubPix(img2, (1, 1), (x2, y2))
    #             stichedImage[y1 - boundaryPoints[0], x1 - boundaryPoints[1]] = pixelValueImg2[0][0]
    #             if (y1 - boundaryPoints[0] < stichedImage.shape[0]) and (
    #                     x1 - boundaryPoints[1] < stichedImage.shape[1]) and (y1 < img1.shape[0]):
    #                 # if y1 < img1.shape[0] :
    #                 print('y1 is ', y1, 'Condition is ', img1.shape[0] - boundaryPoints[0])
    #                 print('y1-boundary[0] ', y1 - boundaryPoints[0])
    #                 # stichedImage[y1 - boundaryPoints[0], x1 - boundaryPoints[1]] = pixelValueImg2[0][0]

    print('-------Exiting the Stitched Image -----------')
    return stichedImage

    # stichedImage[x,y,0] = pixelValue[0][0][0]
    # stichedImage[x,y,1] = pixelValue[0][0][1]
    # stichedImage[x,y,2 ] = pixelValue[0][0][2]

    # print('Stitiched Pixel', stichedImage[x,y])
    # print('Pixel Value', pixelValue[0][0][0])
    # print('Projected points ', x, y, 'Pixel value', pixelValue)


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
