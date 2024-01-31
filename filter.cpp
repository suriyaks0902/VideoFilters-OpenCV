/*
  Saikiran Juttu & Suriya Kasiyalan Siva
  Spring 2024
  01/27/2024
  CS 5330 Computer Vision
*/
#include <cstdio>                    // This is to include C++ input-output libraries
#include <iostream>
#include <opencv2/opencv.hpp>        // This header covers all the OpenCV functions 
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
using namespace cv;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Customgreyscale Function
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Function to convert a 3-channel image to custom grayscale
static int Customgreyscale(cv::Mat& src, cv::Mat& dst) {
    CV_Assert(src.type() == CV_8UC3); // Ensure input is a 3-channel image
    // Create a destination matrix with the same size and type as the source
    dst.create(src.size(), src.type());
    // Loop through each row of the source matrix
    for (int i = 0; i < src.rows; ++i) {
        // Loop through each column of the source matrix
        for (int j = 0; j < src.cols; ++j) {
            // Get the pixel value at (i, j) in the source matrix
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(255 - pixel[2], 255 - pixel[2], 255 - pixel[2]);
        }
    }

    return 0; // Return 0 on success
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  applySepiaTone Function
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Function to apply Sepia tone filter to a pixel
static int applySepiaTone(cv::Mat& src, cv::Mat& dst) {
    CV_Assert(src.type() == CV_8UC3); // Ensure input is a 3-channel image

    dst.create(src.size(), src.type());
    // Iterate over each pixel and apply the Sepia tone filter
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            // Get the original color values
            int red = pixel[2];
            int green = pixel[1];
            int blue = pixel[0];

            // Calculate new color values using the Sepia matrix
            int newRed = cv::saturate_cast<uchar>(0.393 * red + 0.769 * green + 0.1869 * blue);
            int newGreen = cv::saturate_cast<uchar>(0.349 * red + 0.686 * green + 0.168 * blue);
            int newBlue = cv::saturate_cast<uchar>(0.272 * red + 0.534 * green + 0.131 * blue);

            // Update the pixel with modified color values
            pixel[2] = newRed;    // Red
            pixel[1] = newGreen;  // Green
            pixel[0] = newBlue;   // Blue
            dst.at<cv::Vec3b>(i, j) = pixel;
        }

    }

    return 0;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  blur5x5_2 Function
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static int blur5x5_2(cv::Mat& src, cv::Mat& dst) {
    // Copy the source image to the destination image
    dst = src.clone();

    // Get image dimensions
    int rows = src.rows;
    int cols = src.cols;

    // Separable 1x5 filters for vertical and horizontal blur
    int kernel[5] = { 1, 2, 4, 2, 1 };

    // Apply the horizontal blur filter
    for (int i = 0; i < rows; ++i) {
        for (int j = 2; j < cols - 2; ++j) {
            for (int c = 0; c < src.channels(); ++c) {
                int sum = 0;
                // Convolution with the 1x5 kernel horizontally
                for (int n = -2; n <= 2; ++n) {
                    sum += src.ptr<cv::Vec3b>(i)[j + n][c] * kernel[n + 2];
                }
                // Update the destination pixel
                dst.ptr<cv::Vec3b>(i)[j][c] = sum / 10;  // Integer approximation of Gaussian
            }
        }
    }

    // Apply the vertical blur filter
    for (int i = 2; i < rows - 2; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int c = 0; c < src.channels(); ++c) {
                int sum = 0;
                // Convolution with the 1x5 kernel vertically
                for (int m = -2; m <= 2; ++m) {
                    sum += dst.ptr<cv::Vec3b>(i + m)[j][c] * kernel[m + 2];
                }
                // Update the destination pixel
                dst.ptr<cv::Vec3b>(i)[j][c] = sum / 10;  // Integer approximation of Gaussian
            }
        }
    }

    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  blurQuantize Function
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static int blurQuantize(cv::Mat& src, cv::Mat& dst) {
    // Apply Gaussian blur to the input image
    cv::Mat blurred;
    cv::GaussianBlur(src, blurred, cv::Size(5, 5), 0);
    int levels = 10;
    // Quantize each color channel into 'levels' values
    float bucketSize = 255.0 / levels;

    // Iterate through each pixel
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            // Iterate through each color channel (BGR)
            for (int c = 0; c < src.channels(); c++) {
                // Quantize the color value
                float originalValue = blurred.at<cv::Vec3b>(y, x)[c];
                float quantizedValue = std::round(originalValue / bucketSize) * bucketSize;

                // Update the pixel value
                dst.at<cv::Vec3b>(y, x)[c] = static_cast<uchar>(quantizedValue);
            }
        }
    }

    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  detectFaces Function
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Function to detect faces
int detectFaces(cv::Mat &grey, std::vector<cv::Rect> &faces) {
        static cv::Mat half;
        static cv::CascadeClassifier face_cascade;
        static cv::String face_cascade_file(FACE_CASCADE_FILE);

        if (face_cascade.empty()) {
            if (!face_cascade.load(face_cascade_file)) {
                printf("Unable to load face cascade file\n");
                printf("Terminating\n");
                exit(-1);
            }
        }

        faces.clear();
        cv::resize(grey, half, cv::Size(grey.cols / 2, grey.rows / 2));
        cv::equalizeHist(half, half);
        face_cascade.detectMultiScale(half, faces);
            /* std::cout <<"*****************inside df********" << std::endl;
              print statements for debugging 
           */

        for (int i = 0; i < faces.size(); i++) {
            faces[i].x *= 2;
            faces[i].y *= 2;
            faces[i].width *= 2;
            faces[i].height *= 2;
        }

        return 0;
}    
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  brightness_contrast Function
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static int brightness_contrast(cv::Mat& frame, cv::Mat& processedFrame) {
    using namespace cv;
    using namespace std;
    if (frame.empty()) {
        cout << "Could not open or find the webcam stream!" << endl;
        return -1;
    }
    int alpha = 0;
    int beta = 0;
    int alpha_slider = 50;
    int beta_slider = 50;

    createTrackbar("Contrast (Alpha)", "Processed Frame", &alpha_slider, 100);

    createTrackbar("Brightness (Beta)", "Processed Frame", &beta_slider, 100);

    while (true) {
        alpha = static_cast<double>(alpha_slider) / 50.0;

        beta = beta_slider - 50;

        //applyFilter(frame, processedFrame, alpha, beta);
        processedFrame = Mat::zeros(frame.size(), frame.type());

        for (int y = 0; y < frame.rows; y++) {
            for (int x = 0; x < frame.cols; x++) {
                for (int c = 0; c < frame.channels(); c++) {
                    processedFrame.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(alpha * frame.at<Vec3b>(y, x)[c] + beta);
                }
            }
        }
    }

    return 0;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SOBEL X FILTER
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Function to apply Sobel X filter to a 3-channel image
static int sobelX3x3(cv::Mat& src, cv::Mat& dst) {
    dst.create(src.size(), CV_16SC3);

    for (int x = 1; x < src.rows - 1; ++x) {
        for (int y = 1; y < src.cols - 1; ++y) {
            for (int k = 0; k < 3; ++k) {
                // Sobel X filter calculation
                int gx = src.at<cv::Vec3b>(x - 1, y + 1)[k] - src.at<cv::Vec3b>(x - 1, y - 1)[k]
                    + 2 * (src.at<cv::Vec3b>(x, y + 1)[k] - src.at<cv::Vec3b>(x, y - 1)[k])
                    + src.at<cv::Vec3b>(x + 1, y + 1)[k] - src.at<cv::Vec3b>(x + 1, y - 1)[k];
                dst.at<cv::Vec3s>(x, y)[k] = static_cast<short>(gx);
            }
        }
    }

    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SOBEL Y FILTER
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Function to apply Sobel Y filter to a 3-channel image
static int sobelY3x3(cv::Mat& src, cv::Mat& dst) {
    dst.create(src.size(), CV_16SC3);

    for (int x = 1; x < src.rows - 1; ++x) {
        for (int y = 1; y < src.cols - 1; ++y) {
            for (int k = 0; k < 3; ++k) {
                // Sobel Y filter calculation
                int gy = src.at<cv::Vec3b>(x + 1, y - 1)[k] - src.at<cv::Vec3b>(x - 1, y - 1)[k]
                    + 2 * (src.at<cv::Vec3b>(x + 1, y)[k] - src.at<cv::Vec3b>(x - 1, y)[k])
                    + src.at<cv::Vec3b>(x + 1, y + 1)[k] - src.at<cv::Vec3b>(x - 1, y + 1)[k];
                dst.at<cv::Vec3s>(x, y)[k] = static_cast<short>(gy);
            }
        }
    }

    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GRADIENT MAGNITUDE FILTER
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Function to calculate the gradient magnitude of two images
static int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst) {
    if (sx.size() != sy.size()) {
        std::cerr << "Error: Input images must have the same size." << std::endl;
        return -1;
    }

    dst = cv::Mat(sx.size(), CV_8UC3);

    for (int y = 0; y < sx.rows; ++y) {
        for (int x = 0; x < sx.cols; ++x) {
            for (int k = 0; k < 3; ++k) {
                // Calculate the gradient magnitude
                double mag = cv::norm(cv::Vec2d(sx.at<cv::Vec3s>(y, x)[k], sy.at<cv::Vec3s>(y, x)[k]));
                dst.at<cv::Vec3b>(y, x)[k] = static_cast<uchar>(cv::saturate_cast<uchar>(mag));
            }
        }
    }

    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// STRONG COLOR AND REST GRAY SCALE
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Function to enhance strong color in an image based on hue value and range.
// Parameters:
//   - frame: Input BGR image
//   - hueValue: Hue value around which to identify strong color
//   - hueRange: Range of acceptable hue values for strong color identification
static void strongColor(cv::Mat& frame, int hueValue, int hueRange) {
    // Convert the input frame from BGR to HSV color space
    cv::Mat hsvFrame;
    cv::cvtColor(frame, hsvFrame, cv::COLOR_BGR2HSV);

    // Create a binary mask to isolate pixels with strong color based on hue
    cv::Mat mask;
    cv::inRange(hsvFrame, cv::Scalar(hueValue - hueRange, 100, 100), cv::Scalar(hueValue + hueRange, 255, 255), mask);
    
    // Isolate the colored part of the original frame using the mask
    cv::Mat coloredPart;
    cv::bitwise_and(frame, frame, coloredPart, mask);

    // Convert the non-colored part to grayscale
    cv::Mat grayPart;
    cv::cvtColor(frame, grayPart, cv::COLOR_BGR2GRAY);
    cv::cvtColor(grayPart, grayPart, cv::COLOR_GRAY2BGR);

    // Combine the colored and grayscale parts to get the enhanced image
    cv::Mat maskInv = ~mask; // Invert the mask
    cv::Mat grayPartMasked;
    cv::bitwise_and(grayPart, grayPart, grayPartMasked, maskInv);
    cv::add(coloredPart, grayPartMasked, frame);
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////// EXTENSION ///////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Vignette filter
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Function to apply a vignette effect to an input image.
// Parameters:
//   - src: Input image (BGR)
//   - vignetteStrength: Strength of the vignette effect, controlling the fading intensity
static void applyVignette(cv::Mat& src, double vignetteStrength) {
    // Get the width and height of the input image
    int width = src.cols;
    int height = src.rows;

    // Loop through each pixel in the image
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Access the RGB values of the current pixel
            cv::Vec3b& pixel = src.at<cv::Vec3b>(y, x);
            
            // Calculate the vignetting effect with faded curves based on pixel position
            double vignette = pow((1.0 - sqrt(pow((x - width / 2.0) / (width / 2.0), 4) + pow((y - height / 2.0) / (height / 2.0), 4))), vignetteStrength);
          
            // Apply the vignetting effect to each color channel individually
            pixel[2] *= vignette; // Red channel
            pixel[1] *= vignette; // Green channel
            pixel[0] *= vignette; // Blue channel
        }
    }
}
