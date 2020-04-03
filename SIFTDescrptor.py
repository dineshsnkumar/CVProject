import cv2
import numpy as np
import matplotlib.pyplot as plt
from RANSAC import ransac
from StichImages import *
from FeatureDetectorandExtraction import *



def siftMatches(sift, gray_img1, gray_img2):
    kp1, des1 = sift.detectAndCompute(gray_img1, None)
    kp2, des2 = sift.detectAndCompute(gray_img2, None)

    # Match KeyPoints in both the images
    bfmatcher = cv2.BFMatcher()
    matches = bfmatcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches, kp1, kp2