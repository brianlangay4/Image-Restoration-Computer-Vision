# Image-Restoration-Computer-Vision
 image processing techniques for  eye detection and red-eye removal, which are essential components of image restoration. The context provided by these snippets lays the groundwork for understanding how image restoration algorithms can be implemented and applied in real-world scenarios to enhance the quality of digital images.

![Screenshot 2024-03-04 214443](https://github.com/brianlangay4/Image-Restoration-Computer-Vision/assets/67788456/714097a0-01ab-43dc-86b0-d6cc68d96b97)

## OpenCV is a powerful library for computer vision tasks in Python. It includes tools for image processing, object detection, and more. One of its key features is the Haar cascade classifier, a machine learning-based algorithm used for object detection.

# In the context of eye reduction, OpenCV's Haar cascade classifier is particularly useful. By training on a dataset of positive (eye-containing) and negative (eye-lacking) images, it learns to detect eyes in images. This pre-trained classifier can then be applied to new images to automatically locate eye regions. This functionality is leveraged in tasks like red-eye reduction, where the detected eye regions are processed to remove unwanted red-eye effects.

**Eye processing**

 ```'''
To understand how the code detects and removes red eyes, let's break down the relevant parts:
   ```
1. **Eye Detection**:
   ```python
   eyes = eyesCascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(100, 100))
   ```
   - This line utilizes the Haar cascade classifier (`eyesCascade`) to detect eyes in the input image (`img`). 
   - The `detectMultiScale` function detects objects (in this case, eyes) of different sizes in the input image. It returns a list of rectangles where it believes it found eyes.

2. **Processing Detected Eyes**:
   ```python
   for (x, y, w, h) in eyes:
   ```
   - This loop iterates over each detected eye, represented by its bounding box `(x, y, w, h)`.

3. **Extracting Eye Region**:
   ```python
   eye = img[y:y+h, x:x+w]
   ```
   - This line extracts the region of interest (ROI) from the original image (`img`) corresponding to the detected eye. It crops the image based on the coordinates of the bounding box.

4. **Red Eye Removal**:
   - Once the eye region is extracted, the code performs the following steps to remove red-eye effect:
     - **Extracting Channels**: It separates the eye image into its three color channels: blue (`b`), green (`g`), and red (`r`).
     - **Calculating Background**: It calculates the sum of blue and green channels (`bg`), representing the background color without the red-eye effect.
     - **Creating Mask**: It creates a binary mask (`mask`) to identify pixels that are significantly more red than the background. This is done by comparing the red channel (`r`) to a threshold (150) and ensuring that it's greater than both the background and green channels.
     - **Cleaning Mask**: The mask is cleaned by filling holes and dilating to refine the red-eye regions.
     - **Replacing Red Eye**: The mean color of the background (average of blue and green channels) is calculated and used to replace the red-eye region. This is achieved by applying the mean color where the mask is true, effectively removing the red-eye effect.

5. **Displaying Results**:
   ```python
   cv2.imshow('Red Eyes', img)
   cv2.imshow('Red Eyes Removed', imgOut)
   cv2.waitKey(0)
   ```
   - Finally, the original image with detected red eyes and the processed image with red eyes removed are displayed for comparison.


*Full code*
1. **Importing Libraries**:
   ```python
   import cv2
   import numpy as np
   ```
   - `cv2`: OpenCV library for computer vision tasks.
   - `numpy`: Library for numerical operations in Python.

2. **Define a Function to Fill Holes in a Mask**:
   ```python
   def fillHoles(mask):
       maskFloodfill = mask.copy()
       h, w = maskFloodfill.shape[:2]
       maskTemp = np.zeros((h+2, w+2), np.uint8)
       cv2.floodFill(maskFloodfill, maskTemp, (0, 0), 255)
       mask2 = cv2.bitwise_not(maskFloodfill)
       return mask2 | mask
   ```
   - `fillHoles`: This function takes a mask (binary image) as input and fills holes within objects detected in the mask.
   - `maskFloodfill`: A copy of the input mask.
   - `h, w`: Height and width of the mask.
   - `maskTemp`: Temporary mask for flood fill operation.
   - `cv2.floodFill`: Fills the holes in the mask using flood fill algorithm.
   - `mask2`: Inverts the filled mask to obtain the holes.
   - Finally, returns the combined mask of filled holes and original mask.

3. **Main Function**:
   ```python
   if __name__ == '__main__':
   ```
   - This block ensures that the following code runs only when the script is executed directly, not when it's imported as a module.

4. **Read Image and Initialize Variables**:
   ```python
   img = cv2.imread("dv/redeye/red_eyes2.jpg", cv2.IMREAD_COLOR)
   imgOut = img.copy()
   eyesCascade = cv2.CascadeClassifier("dv/haarcascade_eye.xml")
   ```
   - Reads an image "red_eyes2.jpg" and creates a copy.
   - Loads a Haar cascade classifier for eye detection.
   
5. **Detect Eyes**:
   ```python
   eyes = eyesCascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(100, 100))
   ```
   - Uses the Haar cascade to detect eyes in the image.

6. **Process Detected Eyes**:
   ```python
   for (x, y, w, h) in eyes:
   ```
   - Iterates over each detected eye.

7. **Extract Channels and Compute Mask**:
   ```python
   b = eye[:, :, 0]
   g = eye[:, :, 1]
   r = eye[:, :, 2]
   bg = cv2.add(b, g)
   mask = (r > 150) &  (r > bg)
   ```
   - Extracts blue, green, and red channels from the eye.
   - Computes a mask to identify red regions in the eye.

8. **Clean the Mask**:
   ```python
   mask = mask.astype(np.uint8) * 255
   mask = fillHoles(mask)
   mask = cv2.dilate(mask, None, anchor=(-1, -1), iterations=3, borderType=1, borderValue=1)
   ```
   - Converts the mask to uint8 format.
   - Fills holes in the mask and dilates it to refine the regions.

9. **Calculate Mean and Replace Red Eye**:
   ```python
   mean = bg / 2
   mask = mask.astype(bool)[:, :, np.newaxis]
   mean = mean[:, :, np.newaxis]
   eyeOut = np.where(mask, mean, eyeOut)
   ```
   - Calculates the mean of blue and green channels.
   - Replaces the red-eye region with the mean color.

10. **Display Results**:
    ```python
    cv2.imshow('Red Eyes', img)
    cv2.imshow('Red Eyes Removed', imgOut)
    cv2.waitKey(0)
    ```
    - Displays the original image with detected red eyes and the processed image with red eyes removed.
