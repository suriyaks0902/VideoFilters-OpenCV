/*
  Suriya Kasiyalan Siva
  Spring 2024
  01/27/2024
  CS 5330 Computer Vision
  Northeastern University

  Functions for finding faces and drawing boxes around them

  The path to the Haar cascade file is define in faceDetect.h
*/
#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    // Read an image from file
    cv::Mat image = cv::imread("lenna.jpeg");
    //cv::namedWindow("Image Display", cv::WINDOW_AUTOSIZE);
    cv::resize(image, image, cv::Size(1024, 1024));
    // Check if the image is successfully loadedmak
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    // Display the image in a window
    cv::imshow("Image Display", image);

    // Main loop to check for keypress
    while (true) {
        // Wait for a key event
        int key = cv::waitKey(0);

        // Check keypress
        if (key == 'q') {
            break;  // Quit the program if 'q' is pressed
        }
        else {
            // Add other functionality based on different keypresses if needed
            // For example, you can add actions for different keys
            // if (key == 'a') { /* Action for 'a' key */ }
            // else if (key == 'b') { /* Action for 'b' key */ }
        }
    }

    // Close the window
    cv::destroyAllWindows();

    return 0;
}
