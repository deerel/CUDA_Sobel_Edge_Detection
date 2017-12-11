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

void getGaussianKernel(matrix *mat);

void getGxKernel(matrix *mat);

void getGyKernel(matrix *mat);