#pragma once

#include "opencv2\opencv.hpp"
#include "img_structures.h"
#include <stdint.h>

using namespace cv;

extern int16_t maxPixel;
extern int maxPos;

void convertToGrayscale(Mat src, int16_t *dst);

void pixelMatMul(int16_t *src, int16_t *dst, matrix *mat, int width, int height, bool divide = true);

void pixelPyth(int16_t *dst, int16_t *gx, int16_t *gy, const int width, const int height);

void normalize(int16_t *src, const int width, const int height);

int16_t mapToRange(int16_t x, int16_t srcMax, uint16_t dstMax);

void makeImage(int16_t *src, Mat *dst);

void getGaussianKernel(matrix *mat);

void getGxKernel(matrix *mat);

void getGyKernel(matrix *mat);


