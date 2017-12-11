#pragma once
#include <stdint.h>

typedef struct {
	int element[3][3];
} matrix;

int16_t mapToRange(int16_t x, int16_t srcMax, uint16_t dstMax);

void getGaussianKernel(matrix *mat);

void getGxKernel(matrix *mat);

void getGyKernel(matrix *mat);