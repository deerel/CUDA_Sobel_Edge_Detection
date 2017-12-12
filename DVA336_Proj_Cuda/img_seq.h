#pragma once

#include "opencv2\opencv.hpp"
#include "img_structures.h"
#include <stdint.h>

using namespace cv;

extern int16_t maxPixel;
extern int maxPos;

void convertToGrayscale(pixel *src, int16_t *dst, const int len);

void pixelMatMul(int16_t * src, int16_t *dst, matrix * mat, const int width, const int height, bool divide = true);

void pixelPyth(int16_t *dst, int16_t *gx, int16_t *gy, const int width, const int height);

void normalize(int16_t *src, const int width, const int height);

void makeImage(int16_t *src, Mat *dst);

void seq_edge_detection(int16_t *src, Mat * image);



