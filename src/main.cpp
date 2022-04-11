#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <hello.h>
#include <hello.cuh>
#include <chrono>

#define NUM_ITER 200

using namespace std::chrono;
using namespace cv;
using namespace std;
int main(int argc, char** argv) {
    if (argc != 2) {
        cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
        return -1;
    }

  Mat image;
  image = imread(argv[1],0);  // Read the file
  if (image.empty())                      // Check for invalid input
  {
      cout << "Could not open or find the image" << std::endl;
      return -1;
  }
 // namedWindow("Display window",
 //             WINDOW_AUTOSIZE);     // Create a window for display.
 // imshow("Display window", image);  // Show our image inside it.
 // waitKey(0);                       // Wait for a keystroke in the window
	 
  Ptr<SIFT> detector = SIFT::create();
  vector<KeyPoint> keypoints;
  auto start = high_resolution_clock::now();
  for(int i = 0; i < NUM_ITER; i++){ 
    detector->detect(image, keypoints );
  }
  auto end = high_resolution_clock::now();
  auto diff = duration_cast<milliseconds>(end - start);
  cout << "Average time for " << NUM_ITER << " iterations: " << diff.count()/NUM_ITER << "ms" << endl;
  cout << "Image Size (" << image.size().height << "," << image.size().width << ")" << endl;
    //-- Draw keypoints
 // Mat img_keypoints;
 // drawKeypoints(image, keypoints, img_keypoints );
 //   //-- Show detected (drawn) keypoints
 // imshow("SIFT Keypoints", img_keypoints );
 // waitKey();

 // printf("Result of 1+2 is: %d\n", add(1, 2));
 // wrap_hello(10);
  return 0;
}
