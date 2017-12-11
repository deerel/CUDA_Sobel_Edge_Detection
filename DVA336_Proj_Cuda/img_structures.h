#pragma once
#include <stdint.h>

typedef struct {
	int element[3][3];
} matrix;

typedef struct {
	int16_t b;
	int16_t g;
	int16_t r;
} pixel;

int16_t mapToRange(int16_t x, int16_t srcMax, uint16_t dstMax);

void getGaussianKernel(matrix *mat);

void getGxKernel(matrix *mat);

void getGyKernel(matrix *mat);