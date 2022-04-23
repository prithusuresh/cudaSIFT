#ifndef _CUDASIFT_H
#define _CUDASIFT_H
#include <opencv2/core.hpp>

using namespace cv;

#define NUM_OCTAVES_DFL 4
#define NUM_INTERVALS_DFL 2

class cudaSift {
private:
  Mat image;
  int num_octaves;
  int num_intervals;

  unsigned char ***scale_space;

public:
  cudaSift(Mat i, int no, int ni);
  cudaSift(Mat i);
  void init();
  ~cudaSift();
  void describe();
};

#endif //_CUDASIFT_H