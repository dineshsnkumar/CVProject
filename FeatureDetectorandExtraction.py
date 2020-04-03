import cv2 as cv
import numpy as np


def harris_corner_feature_detector(img, detector=False):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Compute Gradients
    kernelX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Ix = cv.Sobel(gray_img,cv.CV_64F, 1,0 , ksize= 5)
    # IY = cv.Sobel(gray_img,cv.CV_64F, 0,1 , ksize= 5)


    Ix = cv.filter2D(gray_img, -1, kernelX)
    Iy = cv.filter2D(gray_img, -1, kernelY)

    # Compute Harris Matrix

    Ixx = np.square(Ix)
    Iyy = np.square(Iy)
    IxIy = Ix * Iy

    # Apply 5X5 kernel on the Ixx , Iyy and IxIY
    gaussian_Ixx = cv.GaussianBlur(Ixx, (5, 5), 0)
    gaussian_Iyy = cv.GaussianBlur(Iyy, (5, 5), 0)
    gaussian_IxIy = cv.GaussianBlur(IxIy, (5, 5), 0)

    # For each pixel calculate the corner strength

    cornerStrength = np.zeros([gray_img.shape[0], gray_img.shape[1]], dtype=np.uint8)

    for x in range(gray_img.shape[0]):
        for y in range(gray_img.shape[1]):
            pixel_Ixx = gaussian_Ixx[x, y]
            pixel_Iyy = gaussian_Iyy[x, y]
            pixel_IxIy = gaussian_IxIy[x, y]

            det = ((pixel_Ixx * pixel_Iyy) - (pixel_IxIy * pixel_IxIy))
            trace = pixel_Ixx + pixel_Iyy
            epsilon = np.finfo(float).eps

            corner = (det) / (trace + epsilon)

            # corner = det - (0.04 * (trace**2))

            cornerStrength[x, y] = corner

    threshold = 0.6 * cornerStrength.max()
    corners = np.zeros([gray_img.shape[0], gray_img.shape[1]], dtype=np.uint8)

    for x in range(cornerStrength.shape[0]):
        for y in range(cornerStrength.shape[1]):
            pixel = cornerStrength[x, y]
            if pixel >= threshold:
                corners[x, y] = cornerStrength[x, y]
                cv.circle(img, (x, y), 6, (0, 255, 0), -1)

    kernel = np.ones((5, 5), np.uint8)
    return corners, img


'''
Given a descriptor calculate the two shortest closing matching features in second Descriptor
Input : feat, feature2
Output : firstShortest, secondShortest

'''
def calculateShortestDistances (feat1, feature2):
    npFeat1 = convertTONumpyArray(feat1)
    ssdDictSSD = {}
    for feat2 in feature2:
        npFeat2 = convertTONumpyArray(feat2)
        ssd = (np.square(npFeat1 - npFeat2).sum())
        ssdDictSSD[ssd] = feat2
    sortValuesAscDict = dict(sorted(ssdDictSSD.items())[:2])
    return sortValuesAscDict


def ratio_test(matches):
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])
    return good_matches

def getGrids(neighborhood):
    window = 4
    subarr = []
    for i in range(0, neighborhood.shape[0] - window + 1, 4):
        for j in range(0, neighborhood.shape[0] - window + 1, 4):
            testWindowij = neighborhood[i:i + window, j:j + window]
            subarr.append(testWindowij)

    listGrids = np.array(subarr)
    return listGrids


def convertTONumpyArray(feature1):
    feature1List = []
    for vect in feature1:
        tempvect = list(vect.values())
        feature1List.append(tempvect)
    npFeature1List = np.array(feature1List)
    return npFeature1List


'''
Given gradient orientation return FeatureVector
'''

def angleToFeatureVector(angle):
    feature_descriptor = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
    }

    for i in range(0, angle.shape[0]):
        for j in range(0, angle.shape[0]):
            value = angle[i, j]
            remainder = (int)(value / 45)
            feature_descriptor[remainder] += 1

    return feature_descriptor


def normalizeFeature(feature_desc):
    # Normalize descriptor for Contrast invariant (d < 0.2)
    numpyFeature= np.array(list(feature_desc.values()))
    totalSum = np.square(numpyFeature).sum()
    threshold = 0.2 * totalSum
    for key, value in feature_desc.items():
         if value > threshold:
                feature_desc[key] = threshold
    return feature_desc

def generateFeatureDescriptor(img1, corners1):
    gray_img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

    descriptor = []
    w = img1.shape[0]
    h = img1.shape[1]

    featureCoordinates = {}

    # 1. Start with 16X16 window
    indexNeighbors1 = np.argwhere(corners1 != 0)
    for intrestPoint in indexNeighbors1:
        x, y = intrestPoint
        # Ignoring Interest points at the boundaries
        if (x >= 8 and y >= 8) and (w - x >= 8 and h - y >= 8):
            # 16 X 16 Window
            neighborhood = gray_img1[x - 8: x + 8, y - 8:y + 8]
            coordinates = (x, y)
            listGrids = getGrids(neighborhood)
            # Orientation Histogram for each cell
            gridDescriptor = []
            for grid in listGrids:
                # Calculate gradient along X and Y axis
                sobelx = cv.Sobel(grid, cv.CV_32F, 1, 0, ksize=3)
                sobely = cv.Sobel(grid, cv.CV_32F, 0, 1, ksize=3)
                magnitude, angle = cv.cartToPolar(sobelx, sobely, angleInDegrees=True)
                feature_desc = angleToFeatureVector(angle)
                norm_feature_desc = normalizeFeature(feature_desc)
                gridDescriptor.append(norm_feature_desc)
            descriptor.append(gridDescriptor)
            featureCoordinates[coordinates] = gridDescriptor

    return descriptor, featureCoordinates



def lookUpCoordinates(feature, cocordianteFeature):
    for item, value in cocordianteFeature.items():
        if value == feature:
            return item

'''
Compute the distance between two features 
------------------------------------------
Given a feature compute the distance between them 
1. SSD 
'''
def findDistanceBetweenDescriptors(feature1, feature2, featureCoordinates, featureCoordinates2):
    ssdDict = {}
    featureKeyPoints = {}
    matchKeyPoints= []
    print('Finding distance between features')
    for feat1 in feature1:
        dictTopTwo = calculateShortestDistances(feat1, feature2)
        threshold = 0.8          # According to paper
        firstDis = list(dictTopTwo.keys())[0]
        secondDis = list(dictTopTwo.keys())[1]
        ratio = firstDis / secondDis
        if ratio <= 0.8:
            featDescrp2 = list(dictTopTwo.values())[0]
            keyPointCoord1 = lookUpCoordinates(feat1, featureCoordinates)
            keyPointCoord2 = lookUpCoordinates(featDescrp2, featureCoordinates2)
            featureKeyPoints[keyPointCoord1] = keyPointCoord2
    return featureKeyPoints, matchKeyPoints


