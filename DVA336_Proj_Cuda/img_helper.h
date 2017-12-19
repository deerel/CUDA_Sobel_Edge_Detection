#pragma once
#include <stdint.h>
#include "img_structures.h"

#ifndef OPENCV
#include "opencv2\opencv.hpp"
#endif

using namespace cv;

int16_t mapToRange(int16_t x, int16_t srcMax, uint16_t dstMax);

void matToArray(Mat * src, pixel *dst);

void compareImages(int16_t * a, int16_t * b, const int elements);