- Harris Corner Detector Not working properly
- Feature Matching

06/02/2020
----------
How to select 4 pair of points ?


08/02/2020
----------
- I found four Points i am getting an error saying that NoneType Object is not iterable
? Should i use secondImage points as the source
? Inlier Threshold (10)
- Once i computed the Homography. I need to project all the first points. How do you compute the second points
? Use H* Point


09/02/2020
----------

-For each match point in the first match point
-I am so lost right now let me backup and understand so what am i doing here 

I found the matches using cv SIFT descriptors.what is the format of the matches 
Matches - [firstImagedescIndex, secondImagedescIndex]
Based on this i found the keyPoints of the images 

kp1[firstImageDescIndex] - first Image Points 
kp2[secondImagedescIndex] - secondImagePoints

I randomly selected 4 matches and found the Homography.

18/02/2020
-----------

todo
- How to test if the curr homography is current ?
- No of inliers 
- NoneType when running ransac



I need to find the inliers for the best homography . Once i find the inliers i need to caluclate
the Homography.

21/02/2020
-------------

I am not sure if the dimensions of the stitched Image is correct. Because the projected points have negitive dimensions
and it is different from the AllStiched Image.

ToDo
---
Text dhaval


---------------------------------------------------------------------------------
22/02/2020
-------------

- I am unsure about the size of the stitched Image. The size of the stitched Image seems to be 
376X235 

ToDo
-----


--------------------------------------------------------------------------------
23/02/2020
-----------

1. What is the problem ?
    - The dimensions of the Stiched Image seems to be different 

------------------------------------------------------------------

24/02/2020
-----------

- The projected Points coordinates is it in x, y or y,x 
- How does numpy store x, y coordinates

-------

--------------------------------------------------------------------

Stitching Images 

1. Copy image1 using the boundary conditions and changing w, h 
2. See if you merge the two images 
 

 29/02/2020
 ------------
 
 - Check if y corordinates is less than img1.shape[0]- boundary[0]
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 















---------------------------------------------------------------------------------



