import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
'''
Project the point (x1, y1) using H
@:argument (x1,y1) , H
@:return (x2, y2)
'''
def project(x1, y1, H):
    a_pers = np.array([[[x1, y1]],
              [[0, 0]]], dtype=np.float64)

    vector = np.array([x1, y1,1], dtype=np.float64)
    proj_trans = np.dot(H, vector)
    proj_coordinates = np.divide(proj_trans, proj_trans[2])
    # print(' Projected Coordinates ', proj_coordinates)
    # print(proj_trans, 'third value is', proj_trans[2])
    # out_pts = cv2.perspectiveTransform(a_pers, H)
    # print('Output from perspective transform', out_pts)
    return proj_coordinates[0], proj_coordinates[1]

'''
Count the no of inliers given the Homography
@:argument Homography, matches, threshold
@:return  totalInliers
'''
def computeInlierCount(H, matches, numMatches,inlierThreshold, kp1, kp2):
    totalNoOfInliers = 0
    inLierMatchesList = []
    # ProjectedPoints = Project the first point in each match using project()
    # ProjectedPoints- OriginalPoints i.e distance < inlierThreshold ----> totalNoOfInliers
    for match in matches:
        ptA = np.float32(kp1[match.queryIdx].pt)
        ptsB = np.float32(kp2[match.trainIdx].pt)

        x1, y1 = ptA[0], ptA[1]
        x2, y2 = ptsB[0], ptsB[1]
        proj_x2, proj_y2 = project(ptA[0], ptA[1], H)
        distance = np.sqrt((x2-proj_x2)**2 + (y2-proj_y2)**2)
        if distance <= inlierThreshold:
            inLierMatchesList.append(match)
            totalNoOfInliers = totalNoOfInliers+1
    print('total no of inLiers ', totalNoOfInliers)
    return totalNoOfInliers, inLierMatchesList

'''
RANSAC
Input : Matches , iterations, threshold
Output : Homography , InverseHomography
'''
def ransac(matches, numMatches, numIterations, inlierThreshold, img1, img2, kp1, kp2):
    print('-----------RANSAC-------------')
    bestInlier = 0
    bestHomography = 0

    # Test Ransac points
    ptsA = np.float32([kp1[match.queryIdx].pt for match in matches])
    ptsB = np.float32([kp2[match.trainIdx].pt for match in matches])

    hom, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, inlierThreshold)
    print('Homography from RANSAC', hom)

    for i in range(0, numIterations):
        fourMatches = random.sample(matches, numMatches)
        # Find points
        ptsA = np.float32([kp1[match.queryIdx].pt for match in fourMatches])
        ptsB = np.float32([kp2[match.trainIdx].pt for match in fourMatches])

        # Compute Homography
        curr_hom = cv2.findHomography(ptsA, ptsB, 0, 0)
        curr_total_Inliners, curr_inLierMatchesList = computeInlierCount(curr_hom[0], matches, numMatches, 10, kp1, kp2)
        if bestInlier < curr_total_Inliners:
            bestInlier = curr_total_Inliners
            bestHomography = curr_hom[0]

    # print('----Best Homography-------', bestHomography)
    # Recompute Homography
    '''
    1. Find Inliers
    2. Recompute Homography using Inliers
    '''
    inlierMatches, inlierMatchesList = computeInlierCount(bestHomography, matches, numMatches, 10, kp1, kp2)
    # print(' No Inlier Matches', inlierMatches, 'Inlier Matches List', inlierMatchesList)
    inlierPtsA = np.float32([kp1[match.queryIdx].pt for match in inlierMatchesList])
    inlierPtsB = np.float32([kp2[match.trainIdx].pt for match in inlierMatchesList])

    inliers_best_hom = cv2.findHomography(inlierPtsA, inlierPtsB, 0, 0)
    inv_inliers_best_hom = np.linalg.inv(inliers_best_hom[0])

    print('inlierPtsA', inlierPtsA,'\n inlierPtsB ', inlierPtsB,'\n-----Best Homography using inliers', inliers_best_hom[0])
    print('\n Inverse Hom', inv_inliers_best_hom)
    print('\n ------RANSAC Completed ------------------')

    return inliers_best_hom, inv_inliers_best_hom, inlierMatchesList