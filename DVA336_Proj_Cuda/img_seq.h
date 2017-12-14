#pragma once

#include "opencv2\opencv.hpp"
#include "img_structures.h"
#include <stdint.h>

using namespace cv;

extern int16_t maxPixel;
extern int maxPos;

void convertToGrayscale(pixel *src, int16_t *dst, const int len);

//void gaussianBlur(int16_t * src, int16_t *dst, matrix * mat, const int width, const int height);

//void sobel(int16_t * src, int16_t *dst, matrix * mat, const int width, const int height);

void gaussianBlur(int16_t * src, int16_t *dst, const int width, const int height);

void sobel_gx(int16_t * src, int16_t *dst, const int width, const int height);

void sobel_gy(int16_t * src, int16_t *dst, const int width, const int height);

void pixelPyth(int16_t *dst, int16_t *gx, int16_t *gy, const int width, const int height);

void normalize(int16_t *src, const int width, const int height);

void makeImage(int16_t *src, Mat *dst);

void getMaxPixel(int16_t * src, const int elements);

void seq_edge_detection(int16_t *src, pixel * pixel_array, const int width, const int height);





