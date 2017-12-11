#pragma once
#include <stdint.h>
#include "img_structures.h"
#include "opencv2\opencv.hpp"

using namespace cv;

int16_t mapToRange(int16_t x, int16_t srcMax, uint16_t dstMax);

void matToArray(Mat * src, pixel *dst);
