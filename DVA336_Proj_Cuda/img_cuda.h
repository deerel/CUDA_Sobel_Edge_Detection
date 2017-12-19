#ifndef STRUCTURES
#include "img_structures.h"
#endif

#ifndef OPENCV
#include "opencv2\opencv.hpp"
#endif

using namespace cv;

void cuda_edge_detection(int16_t * src, pixel * pixel_array, const int width, const int height);

void init_cuda();