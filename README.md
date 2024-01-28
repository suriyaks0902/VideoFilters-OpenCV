# Video Special Effects using OpenCV in C++

## Overview

This GitHub repository contains code for implementing various video special effects using OpenCV in C++. The project focuses on foundational image and video processing techniques, organized into separate files for modularity and clarity.

## Project Structure

- **imgDisplay.cpp:** Handles image display.
- **vidDisplay.cpp:** Extends image display functionality to live video, introducing key commands for user interaction.
- **filter.cpp:** Implements advanced image processing techniques and custom filters.
- **showFaces.cpp, faceDetect.cpp, faceDetect.h:** Integrates face detection into the video stream.

## Features

1. **Live Video Display:**
   - Displays live video.
   - Supports key commands for user interaction.

2. **Advanced Image Processing Techniques:**
   - Greyscale modes for live video.
   - Custom greyscale filter for unique transformations.
   - Sepia tone filter for an antique camera effect.
   - 5x5 blur filter, initially naive and later optimized for speed.
   - Sobel X and Sobel Y filters, with gradient magnitude calculation.

3. **Color Image Manipulation:**
   - Blurring and quantizing color images.
   - User-controlled quantization levels.

4. **Face Detection:**
   - Integrated face detection using provided code files.

5. **Additional Effects:**
   - Strong color preservation and grayscale for the rest.
   - User-adjustable brightness or contrast.
   - Image blurring outside of detected faces.

## Results

### Examples:

- **Greyscale Live Video:**
  - Original Image
  - cvtColor Version of the Greyscale Image

- **Alternative Greyscale Live Video:**
  - Customized Greyscale Image

- **Sepia Tone Filter:**
  - Sepia Tone Filter Image

- **Blur Filters:**
  - Blur Using 2D Gaussian Filter
  - Blur Using Separable 1D Filters

- **Blur and Quantize a Color Image:**
  - Original Image
  - Blur Quantize Image

- **Face Detection Filter:**
  - Image Showing Detected Faces

- **Sobel X and Y Images:**
  - Sobel Gradient Magnitude

- **Adjust Brightness or Contrast:**
  

- **Strong Color Preservation:**
  

### Extensions:

- **Vignetting to Sepia Tone Filter:**
  - Combining vignetting with a sepia tone filter creates a nostalgic and vintage aesthetic.

## Short Reflection

Throughout this project, I gained a comprehensive understanding of computer vision concepts and practical applications using OpenCV in C++. Key takeaways include handling image and video data, implementing diverse filters, and integrating interactive features.

The modular structure emphasized the importance of code organization and comments, facilitating better maintainability and readability. This project significantly enhanced our proficiency in computer vision, bolstering confidence in applying image processing techniques to solve real-world problems.

## Acknowledgement

1. OpenCV Documentation
2. CPP Reference
3. OpenCV Tutorials
4. Stack Overflow Community

Feel free to explore the code and contribute to further enhancements!
