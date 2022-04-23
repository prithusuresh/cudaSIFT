#include "cudaSift.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdlib.h>

using namespace cv;
using namespace std;

cudaSift::cudaSift(Mat i, int no, int ni) {
  image = i;
  num_octaves = no;
  num_intervals = ni;

  init();
}
cudaSift::cudaSift(Mat i)
    : num_octaves(NUM_OCTAVES_DFL), num_intervals(NUM_INTERVALS_DFL) {
  image = i;

  init();
}
void cudaSift::init() {
  scale_space =
      (unsigned char ***)malloc(sizeof(unsigned char **) * num_octaves);

  for (int i = 0; i < num_octaves; i++) {
    scale_space[i] =
        (unsigned char **)malloc(sizeof(unsigned *) * (num_intervals + 3));
  }
}

cudaSift::~cudaSift() {
  for (int i = 0; i < num_octaves; i++) {
    free(scale_space[i]);
  }
  free(scale_space);
}

void cudaSift::describe() {

  cout << "IMAGE DIMENSIONS: (" << image.size().height << ","
       << image.size().width << "," << image.channels() << ")" << endl;
  cout << "Num octaves: " << num_octaves << endl;
  cout << "Num intervals: " << num_intervals << endl;
  cout << "Is Continuous?" << image.isContinuous() << endl;
  imshow("sift source image", image);
  waitKey(0);
}