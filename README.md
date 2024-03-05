# Image-Restoration-Computer-Vision
 image processing techniques for  eye detection and red-eye removal, which are essential components of image restoration. The context provided by these snippets lays the groundwork for understanding how image restoration algorithms can be implemented and applied in real-world scenarios to enhance the quality of digital images.

![Screenshot 2024-03-04 214443](https://github.com/brianlangay4/Image-Restoration-Computer-Vision/assets/67788456/714097a0-01ab-43dc-86b0-d6cc68d96b97)


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
