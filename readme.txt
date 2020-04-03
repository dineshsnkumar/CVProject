Steps in the Program 
-----------------------------------------------

1. Read the Rainer1 , Rainer2 image 
2. Convert to gray scale 
3. Use Harris corner detector to detect the corners and return the corners and write to folder (Output)





Panaroma Stitching for two images 
---------------------
1. Initalize SIFT()
2. Pass the sift, gray_scale1, gray_scale2 to method siftMatches() and get keypoints and matches 
3. Pass matches, kp1, kp2 to ransac to get homography, inverse Homography
4. Pass Images to stitch() to stitch the images 
5. Save the stitched image in output folder "4.png"


-----------------------------------------------------------------------------------------


Panaroma Stitching for Multiple images 
---------------------
1. Stitch 3,4 first 
2. Stitch 3,4,5  (Image A)
3. Stitch 2, 6 
4. Stitch 2,6 with 1 (Image B)
5. Stitch ImageB with ImageA -> Final Image 


-------------------------------------------

Images taken 






