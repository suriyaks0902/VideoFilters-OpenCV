/*
  Saikiran Juttu & Suriya Kasiyalan Siva
  Spring 2024
  01/27/2024
  CS 5330 Computer Vision
*/
#include <opencv2/opencv.hpp>
#include "filter.h"
#include "filter.cpp"
#include "faceDetect.h"

int main(int argc, char* argv[]) {
    // Initialize Video Capture
    cv::VideoCapture* capdev;

    // Open the video device (camera)
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return (-1);
    }

    // Get video frame properties
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    // Create a named window for video display
    cv::namedWindow("Video", 1);

    // Mat object to store each video frame
    cv::Mat frame;
    char lastKey = 'c'; // Default to color display

    // Main processing loop
    for (;;) {
        *capdev >> frame; // Capture a new frame from the camera

        // Check if the frame is empty
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }

        // Apply different filters based on user input
        if (lastKey == 'g') {
            // Convert the frame to grayscale
            cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        }
        if (lastKey == 's') {
            // Save the current frame as an image
            cv::imwrite("image_saved.jpg", frame);
        }
        if (lastKey == 'h') {
            // Apply custom grayscale filter
            Customgreyscale(frame, frame);
        }
        if (lastKey == 't') {
            // Apply sepia tone filter
            applySepiaTone(frame, frame);
        }
        if (lastKey == 'i') {
            // Apply blur and quantize filter
            blurQuantize(frame, frame);
        }
        if (lastKey == 'b') {
            // Apply a 5x5 blur filter
            blur5x5_2(frame, frame);
        }
        if (lastKey == 'a') {
            // Apply brightness and contrast adjustment
            brightness_contrast(frame, frame);
        }
        else if (lastKey == 'f') {
            // Apply face detection filter when 'f' is pressed
            filter::applyFaceDetection(frame);
            cv::imshow("Video", frame);
        
        }
        if (lastKey == 'x') {
            cv::Mat sobelXImage, displayXImage;

            sobelX3x3(frame, sobelXImage);
            // Convert the result of the Sobel X operation (sobelXImage) to absolute values
            // and store the result in displayXImage. This step is often necessary when working
            // with gradient images to ensure that negative values are represented as positive
            // values, making the image suitable for display and visualization.
            cv::convertScaleAbs(sobelXImage, displayXImage);
            cv::imshow("Video1", displayXImage);

            cv::imwrite("Video1.jpg", displayXImage);


        }
        if (lastKey == 'y') {
            cv::Mat sobelYImage, displayYImage;
            sobelY3x3(frame, sobelYImage);
            // Convert the result of the Sobel Y operation (sobelXImage) to absolute values
            // and store the result in displayYImage. This step is often necessary when working
            // with gradient images to ensure that negative values are represented as positive
            // values, making the image suitable for display and visualization.
            cv::convertScaleAbs(sobelYImage, displayYImage);
            cv::imshow("Video2", displayYImage);

            cv::imwrite("Video2.jpg", displayYImage);
        }
        // If the last key pressed is 'm', perform the following image processing steps:
        if (lastKey == 'm') {
        // Initialize matrices to store Sobel X, Sobel Y, and magnitude images.
            cv::Mat sobelXImage, sobelYImage, magnitudeImage;

            // Apply Sobel X and Sobel Y operations to the input frame to compute gradients.
            sobelX3x3(frame, sobelXImage);
            sobelY3x3(frame, sobelYImage);

            // Calculate the magnitude of the gradients using the Sobel X and Sobel Y results.
            magnitude(sobelXImage, sobelYImage, magnitudeImage);

            // Convert the magnitude image to absolute values and scale it for display.
            cv::convertScaleAbs(magnitudeImage, magnitudeImage);

            // Save the processed magnitude image to a file named "Video3.jpg".
            cv::imwrite("Video3.jpg", magnitudeImage);
        }
        else if (lastKey == 'w') {
        // Call the isolateColor function with the desired hue value and range
                 filter::strongColor(frame, 60, 10); // Example for green color
                 cv::imshow("Video", frame);
        }
        else if (lastKey == 'v'){
                 // Adjust vignette strength (reduce to half)
                 double vignetteStrength = 0.15;

                 // Apply the vignette effect
                 applyVignette(frame, vignetteStrength);
                 // Show the output
                 cv::imshow("Webcam with Vignette", frame);
        }
        
        // Display the processed frame
        cv::imshow("Video", frame);

        // Check for a waiting keystroke
        char key = cv::waitKey(10);

        // User input handling
        if (key == 'q') {
            // Exit the program if 'q' is pressed
            break;
        } else if (key == 'g' || key == 'c' || key == 'h' || key == 't' || key == 'b' || key == 'i' || key == 's' || key == 'a'|| key == 'x' || key == 'y' || key == 'm'|| key == 'w'|| key == 'v') {
            // Update lastKey based on user input
            lastKey = key;
        }
    }

    // Release resources
    delete capdev;
    return (0);
}
