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

    # Find the minimum and maximum along y and x columns



    # cv2.rectangle(img1, (int(x1), int(y1)), (int(x3), int(y3)), (255,0,0), 10)
    # cv2.imshow('', img1)
    # cv2.waitKey()

    # img2points1 = [[x1, y1]]
    # img2points2 = [[x2, y2]]
    # img2points3 = [[x3, y3]]
    # img2points4 = [[x4, y4]]

    # points2 = np.array([img2points1, img2points2, img2points3, img2points4], np.float32)
    # print('----Points2', img2points1, img2points2, img2points3, img2points4)


    points = np.array([img1points1, img1points2, img1points3, img1points4, img2points1, img2points2, img2points3, img2points4], np.float32)
    min__points = np.amin(points, axis=0)
    max_points = np.amax(points, axis=0)
    print('Points', points, 'Min and Max along axis', min__points, max_points)
    # h = int(max_points[0][0] - min__points[0][0])
    # w = int(max_points[0][1] - min__points[0][1])

    # print('height, width of stiched Image',h, w)
    # # print('Points ', points)
    boundaryPoints = cv2.boundingRect(points)
    #
    # # w = boundaryPoints[2] - boundaryPoints[0]
    # # h = boundaryPoints[3] - boundaryPoints[1]
    #
    h = boundaryPoints[2] - boundaryPoints[0]
    w = boundaryPoints[3] - boundaryPoints[1]
    stichedImage = np.zeros([h, w, 3], np.uint8)

    print('Image 1 dimensions', img1.shape, 'Image2 dimensions', img2.shape)
    print('Image 1 boundary points', img1points1, img1points2, img1points3, img1points4)
    print('Image 2 boundary', (0,0), (w2, 0), (0, h2), (w2, h2))
    print('Projected Point', img2points1,img2points2, img2points3 ,img2points4)
    print('Boundary Points', boundaryPoints)
    print('Stitched Image', stichedImage.shape)

    # Copy img1 to stitchedImage
    # ? What is the right location ?
    offsetY = stichedImage.shape[0] - img1.shape[0]
    offsetX = stichedImage.shape[1] - img1.shape[1]
    print('OffSet',offsetX, offsetY)

    # Try changing this
    # stichedImage[0:img1.shape[0], 0:img1.shape[1]] = img1
    for i in range(0, img1.shape[0]):
        for j in range(0, img1.shape[1]):
            stichedImage[i - boundaryPoints[0], j - boundaryPoints[1]] = img1[i, j]

    # Project the stitchedImage in image2 space
    '''
    If the point lies in the image2 space 
    add/ blend pixel value to stitched Image 
    ? i,j is y, x 
    '''
    for i in range(boundaryPoints[0], stichedImage.shape[0]):
        for j in range(boundaryPoints[1], stichedImage.shape[1]):
            x, y = project(i,j, H[0])
            # Remove
            if 0 <= x < img2.shape[0] and 0 <= y < img2.shape[1]:
                if (x - boundaryPoints[0] < img2.shape[1]) and (y - boundaryPoints[1] < img2.shape[0]):
                    x = round(x)
                    y = round(y)
                    pixelValue = cv2.getRectSubPix(img2, (1, 1), (x, y))
                    print('j, i', j, i)
                    stichedImage[j - boundaryPoints[0], i - boundaryPoints[1]] = pixelValue[0][0]
                    print('Boundary Points', stichedImage[j - boundaryPoints[0], i - boundaryPoints[1]])
                    print('Pixel Value', pixelValue[0][0])

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





