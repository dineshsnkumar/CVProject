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

    corners, corner_box_img = harris_corner_feature_detector(box_img, False)
    corners1, corner_box_img1 = harris_corner_feature_detector(img7)
    corners2, corner_box_img2 = harris_corner_feature_detector(img8)

    cv2.imwrite('./output/1a.png', corner_box_img)
    cv2.imwrite('./output/1b.png', corner_box_img1)
    cv2.imwrite('./output/1c.png', corner_box_img2)


    # Part 2 -----------FEATURE MATCHING------------------------------------
    descriptor1, featureCoordinates1 = generateFeatureDescriptor(img7, corners1)
    descriptor2, featureCoordinates2 = generateFeatureDescriptor(img8, corners1)

    keyPointCoordinates = findDistanceBetweenDescriptors(descriptor1, descriptor2, featureCoordinates1, featureCoordinates2)


    #  Part 3 ------------PANORAMA MOSAIC STITCHING - Two Images-----------------------

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

    # ----------------------Stitching my images -------------------------
    print('------------Stitching my images---- ')
    my_img1 = cv2.imread('./my_images/1.jpg')
    my_img2 = cv2.imread('./my_images/2.jpg')
    my_img3 = cv2.imread('./my_images/3.jpg')

    # Convert to gray scale
    my_gray_img1 = cv2.cvtColor(my_img1, cv2.COLOR_BGR2GRAY)
    my_gray_img2 = cv2.cvtColor(my_img2, cv2.COLOR_BGR2GRAY)
    my_gray_img3 = cv2.cvtColor(my_img3, cv2.COLOR_BGR2GRAY)

    matches13, kp13, kp3 = siftMatches(sift, my_gray_img1, my_gray_img3)
    # Perform Ransac
    hom1, invHom13, inLierMatchesList13 = ransac(matches13, 4, iterations, 10, my_img1, my_img3, kp13, kp3)
    # Merge the images
    stichedImage13 = stitch(my_img1, my_img3, hom1, invHom13)
    cv2.imwrite("./output/my_image_13.png", stichedImage13)

    print('----Stitched  my images 1, 3---- ')
    my_gray_stichedImage13 = cv2.cvtColor(stichedImage13, cv2.COLOR_BGR2GRAY)
    matches123, kp123, kp2 = siftMatches(sift, my_gray_stichedImage13, my_gray_img2)
    # Perform Ransac
    hom, invHom, inLierMatchesList = ransac(matches123, 4, iterations, 10, stichedImage13, my_img2, kp123, kp2)
    # Merge the images
    stichedImage13 = stitch(stichedImage13, my_img2, hom, invHom)
    cv2.imwrite("./output/my_image_123.png", stichedImage13)

    print('------------Stitching my images Completed---- ')


    # ---------------Part 4 Stitching Multiple images------------------------
    '''
    1. Stitch Ranier1, Rainer 2 to Rainer 6 
    2. Stitch Rainer 3 ,4 and 5 
    3. Merge all 
    '''
    img3 = cv2.imread('project_images/Rainier3.png')
    img4 = cv2.imread('project_images/Rainier4.png')
    img5 = cv2.imread('project_images/Rainier5.png')
    img6 = cv2.imread('project_images/Rainier6.png')

    gray_img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    gray_img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    gray_img5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
    gray_img6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
    stichedImage_gray = cv2.cvtColor(stichedImage, cv2.COLOR_BGR2GRAY)

    print('---Started Rainer 3.4----')

    matches34, kp3, kp4 = siftMatches(sift, gray_img3, gray_img4)
    # Perform Ransac
    hom34, invHom34, inLierMatchesList34 = ransac(matches34, 4, iterations, 10, img3, img4, kp3, kp4)
    # Merge the images
    stichedImage34 = stitch(img3, img4, hom34, invHom34)
    print('---Completed  Rainer 3.4----')
    cv2.imwrite("./output/34.png", stichedImage34)

    print('---Stitching Rainer 3,4 with 5-----')

    gray_stitch_img34 = cv2.cvtColor(stichedImage34, cv2.COLOR_BGR2GRAY)
    matches345, kp34, kp5 = siftMatches(sift, gray_stitch_img34, gray_img5)
    # Perform Ransac
    hom345, invHom345, inLierMatchesList345 = ransac(matches345, 4, iterations, 10, stichedImage34, img5, kp34, kp5)
    # Merge the images
    stichedImage345 = stitch(stichedImage34, img5, hom345, invHom345)
    cv2.imwrite("./output/345.png", stichedImage345)

    print('---Completed  Rainer 3.4 and 5----')

    print('---Stitching Rainer 2,6-----')
    matches26, kp26, kp6 = siftMatches(sift, gray_img2, gray_img6)
    # Perform Ransac
    hom2, invHom26, inLierMatchesList26 = ransac(matches26, 4, iterations, 10, img2, img6, kp26, kp6)
    # Merge the images
    stichedImage26 = stitch(img2, img6, hom2, invHom26)
    cv2.imwrite("./output/26.png", stichedImage26)
    print('---Completed  Rainer 2, 6----')

    print('---Stitching Rainer 2,6 with 1 -----')
    gray_stitch_img26 = cv2.cvtColor(stichedImage26, cv2.COLOR_BGR2GRAY)
    matches126, kp126, kp1 = siftMatches(sift, gray_stitch_img26, gray_img1)
    # Perform Ransac
    hom126, invHom126, inLierMatchesList126 = ransac(matches126, 4, iterations, 10, stichedImage26, img1, kp126, kp1)
    # Merge the images
    stichedImage126 = stitch(stichedImage26, img1, hom126, invHom126)
    cv2.imwrite("./output/126.png", stichedImage126)
    print('---Completed  Rainer 2, 6 with 1----')

    print('---Stitching  ALLSTITCHED :) -----')
    gray_stitch_img126 = cv2.cvtColor(stichedImage126, cv2.COLOR_BGR2GRAY)
    gray_stitch_img345 = cv2.cvtColor(stichedImage345, cv2.COLOR_BGR2GRAY)

    matches, kp, kp1 = siftMatches(sift, gray_stitch_img126, gray_stitch_img345)
    # Perform Ransac
    hom, invHom, inLierMatchesList = ransac(matches, 4, iterations, 10, stichedImage126, stichedImage345, kp, kp1)
    # Merge the images
    stichedImage = stitch(stichedImage126, stichedImage345, hom, invHom)
    cv2.imwrite("./output/AllStitched.png", stichedImage)
    print('---Completed ALLSTITCHED :) ----')




if __name__ == '__main__':
    main()
