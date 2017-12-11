#pragma once
#include "opencv2\opencv.hpp"
#include "img_structures.h"

using namespace cv;

void cuda_edge_detection(int16_t * src, Mat * image);