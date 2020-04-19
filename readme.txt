Steps in the Program 
-----------------------------------------------

Main Class is the starting point of the program 

Assignment 2 
-------------------

1. Read the Rainer1 , Rainer2 image 
2. Convert to gray scale 
3. Use harris_corner_feature_detector() to detect the corners and return the corners and write to folder (Output)
4. generateFeatureDescriptor() to generate descriptors for image1 and image2


Project
------------------

Panaroma Stitching for two images 
---------------------
1. Initalize SIFT()
2. Pass the sift, gray_scale1, gray_scale2 to method siftMatches() and get keypoints and matches 
3. Pass matches, kp1, kp2 to ransac to get homography, inverse Homography
4. Pass Images to stitch() to stitch the images 
5. Save the stitched image in output folder "4.png"

-----------------------------------------------------------------------------------------

Panaroma Stitching for my images 
---------------------
1. Read the images 
2. Stitch 1 and 3 in my_images folder -> Stich13
3. Stich13 with 2.png in my_older
4. output  "my_image_123.png"

-----------------------------------------------------------------------------------------

Panaroma Stitching for Multiple images 
---------------------
1. Stitch 3,4 first 
2. Stitch 3,4,5  (Image A)
3. Stitch 2, 6 
4. Stitch 2,6 with 1 (Image B)
5. Stitch ImageB with ImageA and final Image is at output folder "AllStitched.png"


-----------------------------------------------------------------------------------------------------------------------------------------

Folder 
-------

my_images - Images I took
output - where the program outputs the images 
output_results - The images I got while running the program 


Methods 
-----------

RANSAC 
--

project () - projects point x, y using homography and returns the projected point 
computeInlierCount() - Computes the no of inliers
ransac ()  - performs ransac 



StitchImages 
----
stitch() --- Stitches images given the images and Homography and inverseHomography
















