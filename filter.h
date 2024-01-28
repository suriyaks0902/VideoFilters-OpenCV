/*
  Suriya Kasiyalan Siva
  Spring 2024
  01/27/2024
  CS 5330 Computer Vision
  Northeastern University
*/

#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#define FACE_CASCADE_FILE "./haarcascade_frontalface_alt2.xml"



// Function to convert an image to greyscale using a custom method.
static int Customgreyscale(cv::Mat& src, cv::Mat& dst);

// Function to apply a sepia tone filter to an image.
static int applySepiaTone(cv::Mat& src, cv::Mat& dst);

// Function to apply a 5x5 blur filter to an image.
static int blur5x5_2(cv::Mat& src, cv::Mat& dst);

// Function to apply a blur and quantization effect to an image.
static int blurQuantize(cv::Mat& src, cv::Mat& dst);

// Function to highlight faces in an image.
// Prototypes for face detection
int detectFaces(cv::Mat &grey, std::vector<cv::Rect> &faces);
int drawBoxes(cv::Mat &frame, std::vector<cv::Rect> &faces, int minWidth = 50, float scale = 1.0);
int applyFaceDetection(cv::Mat& frame);

// Function to adjust brightness and contrast of an image.
static int Brightness_contrast(cv::Mat& frame, cv::Mat& outputFrame);

// Function to calculate the magnitude of gradients using Sobel operators.
static int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst);

// Function to apply a Sobel X 3x3 filter to an image.
static int sobelX3x3(cv::Mat& src, cv::Mat& dst);

// Function to apply a Sobel Y 3x3 filter to an image.
static int sobelY3x3(cv::Mat& src, cv::Mat& dst);

// Function to apply a Sobel Y 3x3 filter to an image.
static void strongColor(cv::Mat& frame, int hueValue, int hueRange);

// Function to apply a vignette effect to an image.
static void applyVignette(cv::Mat& src, double vignetteStrength)

#endif // FILTER_H

