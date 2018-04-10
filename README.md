## Advanced Lane Finding Project - P4


---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.



### Files Submitted 

This project includes the following files:
I Had to split up my jupyter notebook project to the follow parts due to notebook unexpectedly crashing :
1. Compute Camera Calibration.ipynb 
2. Perspective Transform.ipynb
3. Color Spaces and Gradient Thresholds.ipynb
4. Finding the Lane (Sliding Window Search).ipynb

Directories :
1. output_images : Sample output images for the writeup
2. output_videos : Pipeline Processed output video
3. pickle_data_store : pickle data store to share data between notebooks.
4. README : This writeup 


### Sample Project Outcome
#### CarND | Project 4 - CarND-Advanced-Lane-Finding | Project Video

<a href="https://www.youtube.com/watch?v=kMyRkVBFjZo&feature=youtu.be" target="_blank"><img src="http://img.youtube.com/vi/kMyRkVBFjZo/0.jpg" 
alt="CarND-Advanced-Lane-Finding | Project Video" width="240" height="180" border="10" /></a>


#### CarND | Project 4 - CarND-Advanced-Lane-Finding | Challenge video

<a href="https://www.youtube.com/watch?v=ijTnYbzQZqk&feature=youtu.be" target="_blank"><img src="http://img.youtube.com/vi/ijTnYbzQZqk/0.jpg" 
alt="CarND | Project 4 - CarND-Advanced-Lane-Finding | challenge video" width="240" height="180" border="10" /></a>

#### CarND | Project 4 - CarND-Advanced-Lane-Finding | Harder Challenge video

<a href="https://www.youtube.com/watch?v=pLRka9xzDr0&feature=youtu.be" target="_blank"><img src="http://img.youtube.com/vi/pLRka9xzDr0/0.jpg" 
alt="CarND-Advanced-Lane-Finding | Harder Challenge video" width="240" height="180" border="10" /></a>


[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"


[calibration-image9]: ./output_images/calibration-image9.jpg

[undistor-corrected]: ./output_images/undistor-corrected.jpg	
[undistor-corrected_board]: ./output_images/undistor-corrected_board.jpg	
[undistor-corrected_sample_image]: ./output_images/undistor-corrected_sample_image.jpg
[undistor-original]: ./output_images/undistor-original.jpg	
[undistor-original_board]: ./output_images/undistor-original_board.jpg	
[undistor-original_sample_image]: ./output_images/undistor-original_sample_image.jpg


[combined_threshold-original-0]: ./output_images/combined_threshold-original-0.jpg	
[combined_threshold-original-1]: ./output_images/combined_threshold-original-1.jpg	
[combined_threshold-perspective-original-0]: ./output_images/combined_threshold-perspective-original-0.jpg	
[combined_threshold-perspective-original-1]: ./output_images/combined_threshold-perspective-original-1.jpg	
[combined_threshold-perspective-processed-0]: ./output_images/combined_threshold-perspective-processed-0.jpg	
[combined_threshold-perspective-processed-1]: ./output_images/combined_threshold-perspective-processed-1.jpg	
[combined_threshold-processed-0]: ./output_images/combined_threshold-processed-0.jpg	
[combined_threshold-processed-1]: ./output_images/combined_threshold-processed-1.jpg


[Perspective_Transform_img_points]: ./output_images/Perspective_Transform_img_points.jpg
[Perspective_Transform_unwarped]: ./output_images/Perspective_Transform_unwarped.jpg

[Sliding-Window-Search-original-0]: ./output_images/Sliding-Window-Search-original-0.jpg
[Sliding-Window-Search-processed-1]: ./output_images/Sliding-Window-Search-processed-1.jpg


[Sliding-Window-Skip-Search-original-0]: ./output_images/Sliding-Window-Skip-Search-original-0.jpg
[Sliding-Window-Skip-Search-processed-1]: ./output_images/Sliding-Window-Skip-Search-processed-1.jpg

[Highlighted_Lane_From_Video]: ./output_images/Highlight_lane_From_video.jpg.png
[Sliding_Window_Lane_Width_over_Time]: ./output_images/Sliding_Window_Lane_Width_over_Time.jpg "Lane Width Tracking against sliding window Thresholds"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./1. Compute Camera Calibration.ipynb" 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 


Image Calibration            | 
:-------------------------:|
![alt text][calibration-image9]  |


 (Original)          |    (Undistored)
:-------------------------:|:-------------------------:
![alt text][undistor-original_board]| ![alt text][undistor-corrected_board]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:



 (Original)          |    (Undistored)
:-------------------------:|:-------------------------:
![alt text][undistor-original_sample_image] | ![alt text][undistor-corrected_sample_image]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps  in `3. Color Spaces and Gradient Thresholds.ipynb`).  Here's an example of my output for this step. I used combination of S thresholding, Sobel, Direction and Magnitude Gradient thresholding. 




Combined Thresholds (Original)          |   Combined Thresholds (Processed)
:-------------------------:|:-------------------------:
![alt text][combined_threshold-original-0] | ![alt text][combined_threshold-processed-0]


Combined Thresholds (Original)          |   Combined Thresholds (Processed)
:-------------------------:|:-------------------------:
![alt text][combined_threshold-original-1] | ![alt text][combined_threshold-processed-1]



Combined Thresholds Perspective (Original)          |   Combined Thresholds Perspective (Processed)
:-------------------------:|:-------------------------:
![alt text][combined_threshold-perspective-original-0] | ![alt text][combined_threshold-perspective-processed-0]


Combined Thresholds Perspective (Original)          |   Combined Thresholds Perspective (Processed)
:-------------------------:|:-------------------------:
![alt text][combined_threshold-perspective-original-1] | ![alt text][combined_threshold-perspective-processed-1]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `image_unwarp()`, which appears in  the file `2. Perspective Transform.ipynb` (output_images/Perspective_Transform_unwarped.jpg).  The `image_unwarp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python

src = np.float32([
                  (595 ,450 ),
                  (686 ,450), 
                   (250,682),
                  (1042,682)])
dst = np.float32([(250,0),
                  (w-250,0),
                  (250,682),
                  (w-250,682)])
                  
                  
def image_unwarp(img, src, dst):
    
    undist, mtx, dist = cal_undistort(img, objpoints, imgpoints)
        
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
        
    img_size = (gray.shape[1], gray.shape[0]) 
        
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    
    
    #Inverse Perspective Transform 
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(undist, M, img_size)
    
    return warped, M, Minv

```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 595 ,450   | 250,0       | 
| 686 ,450      | 1030,0      |
| 250,682     | 250,682      |
| 1042,682     | 1030,682      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Identify Image points        |   Perspective Transform
:-------------------------:|:-------------------------:
![alt text][Perspective_Transform_img_points] | ![alt text][Perspective_Transform_unwarped]



[Perspective_Transform_img_points]: ./output_images/Perspective_Transform_img_points.jpg
[Perspective_Transform_unwarped]: ./output_images/Perspective_Transform_unwarped.jpg


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?



#### Fitting Lane lines with a 2nd order Polynomial

The idea here is to identify lane lines pixel by calculating the histogram of the lower half of the image and identify the peaks. and use convolution to iterate through the image bottom-up. Using n-windows with fixed window Hight. 


I created two functions to perform sliding window search perform_lane_sws(binary_warped) and another function with skip histogram search (lane_detect_no_sws) both present in file (4. Finding the Lane (Sliding Window Search). Also, I provided below sample output images from both functions demonestrating sliding window detection for straight and curved lanes.

[References] Code modified from Carnd-Term1 class


```python

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

```



![alt text][image5]



#### Performing Sliding Window Search

Sliding Window Search (Straight)            |   Sliding Window Search (Curved)
:-------------------------:|:-------------------------:
![alt text][Sliding-Window-Search-original-0] | ![alt text][Sliding-Window-Search-processed-1]



#### Skipping Sliding Window Search

Sliding Window Skip Search (Straight)            |   Sliding Window Skip Search (Curved)
:-------------------------:|:-------------------------:
![alt text][Sliding-Window-Skip-Search-original-0]  | ![alt text][Sliding-Window-Skip-Search-processed-1]




#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I created a function measure_curvature3  `4. Finding the Lane (Sliding Window Search).ipynb` That calculate Lane curvatured by fitting the detected lane points to 2nd order polynomial as following :


```python


def measure_curvature3(ploty, left_fit, right_fit, leftx, rightx):
    
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
       
    return left_curverad, right_curverad

```
#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
I implemented this step in `4. Finding the Lane (Sliding Window Search).ipynb` in the function `highlight_detected_lane()`.  Here is an example of my result on a test image:

![alt text][Highlighted_Lane_From_Video]

---

### Pipeline (video)


#### Pipeline Sanity Checks (Reset)

As long as the expected lane width is detected I skip sliding window search. I use both upper and lower boundry thresholds for the lane width to act as Sanity checks and make sure we didn't lose the lanes due to a difficult frame. I provided below a graph of the lane width for the entire video and highlighted both the uppoer and lower thresholds i used. whenever the width exceed the upper threshold or go below the lower thresholds a sliding window search will be initiated. 

Lane Width Overtime against Sliding Window Thresholds            | 
:-------------------------:|
![alt text][Sliding_Window_Lane_Width_over_Time]  |

% frames Processed using  Sliding window Search= 26.09 %
% frames Skipped Sliding window Search= 73.91 %

#### I provided below a link to the final video output.  The pipeline performed reasonably well on the main project video. However I didn't get the expected result with the challenge video or the Harder challenge video. 


Here's a youtube link to my video result

#### CarND | Project 4 - CarND-Advanced-Lane-Finding | Project Video

<a href="https://www.youtube.com/watch?v=kMyRkVBFjZo&feature=youtu.be" target="_blank"><img src="http://img.youtube.com/vi/kMyRkVBFjZo/0.jpg" 
alt="CarND-Advanced-Lane-Finding | Project Video" width="240" height="180" border="10" /></a>


#### CarND | Project 4 - CarND-Advanced-Lane-Finding | Challenge video

<a href="https://www.youtube.com/watch?v=ijTnYbzQZqk&feature=youtu.be" target="_blank"><img src="http://img.youtube.com/vi/ijTnYbzQZqk/0.jpg" 
alt="CarND | Project 4 - CarND-Advanced-Lane-Finding | challenge video" width="240" height="180" border="10" /></a>

#### CarND | Project 4 - CarND-Advanced-Lane-Finding | Harder Challenge video

<a href="https://www.youtube.com/watch?v=pLRka9xzDr0&feature=youtu.be" target="_blank"><img src="http://img.youtube.com/vi/pLRka9xzDr0/0.jpg" 
alt="CarND-Advanced-Lane-Finding | Harder Challenge video" width="240" height="180" border="10" /></a>


Also the video can be downloaded from here a [link to my video result](./project_video.mp4)






---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The image thresholding technique worked well with the main project video, whiched used combination of S thresholding, Sobel, Direction and Magnitude Gradient thresholding. However this technique didn't work will with other Challenge videos.

The current pipeline can be further improved by using average of coefficients of the past nFrames to facilitate smoother lane detection. Also, The perspective transform can be further tuned to allow accurate curvature estimation.

A further improvement can be also made by implementing a sophisticated masking to filter out the road imperfection from the challenge video. And fine tune the combine_threshold function to work with the harder challenge video.



## [References]

1. https://discussions.udacity.com/t/error-when-test-with-challenge-video/505092/2
2. https://discussions.udacity.com/t/curvature-value-seems-small-and-choppy/643472
3. Udacity CarND Term1 :  Advance Lane Finding
