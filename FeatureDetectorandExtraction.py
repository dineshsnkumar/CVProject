import cv2 as cv
import numpy as np


def harris_corner_feature_detector(img):
    print('-----Feature Detector------')
    # img = cv.imread('image_sets/graf/img1.ppm')
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

    cornerStrengthMatrix = np.zeros([gray_img.shape[0], gray_img.shape[1]], dtype=np.uint8)

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

            cornerStrengthMatrix[x, y] = corner

    threshold = 200

    corners_normalize = np.empty(cornerStrengthMatrix.shape, dtype=np.uint8)
    cv.normalize(cornerStrengthMatrix, corners_normalize, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    dst_norm_scaled = cv.convertScaleAbs(corners_normalize)


    for x in range(cornerStrengthMatrix.shape[0]):
        for y in range(cornerStrengthMatrix.shape[1]):
            pixel = cornerStrengthMatrix[x, y]
            if pixel >= threshold:
                # corners_thresh[x, y] = cornerStrengthMatrix[x, y]
                cv.circle(corners_normalize, (y, x), 6, (0, 255, 0), -1)
    kernel = np.ones((5, 5), np.uint8)
    return corners_normalize, corners_normalize


def ratio_test(matches):
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    return good_matches
