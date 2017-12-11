#include "img_structures.h"


int16_t mapToRange(int16_t x, int16_t srcMax, uint16_t dstMax)
{
	float mapVal = ((float)x / (float)srcMax) * (float)dstMax;
	return (int16_t)mapVal;
}

void getGaussianKernel(matrix *mat)
{
	mat->element[0][0] = 1;
	mat->element[0][1] = 2;
	mat->element[0][2] = 1;
	mat->element[1][0] = 2;
	mat->element[1][1] = 4;
	mat->element[1][2] = 2;
	mat->element[2][0] = 1;
	mat->element[2][1] = 2;
	mat->element[2][2] = 1;
}

void getGxKernel(matrix *mat)
{
	mat->element[0][0] = 1;
	mat->element[0][1] = 0;
	mat->element[0][2] = -1;
	mat->element[1][0] = 2;
	mat->element[1][1] = 0;
	mat->element[1][2] = -2;
	mat->element[2][0] = 1;
	mat->element[2][1] = 0;
	mat->element[2][2] = -1;
}

void getGyKernel(matrix *mat)
{
	mat->element[0][0] = 1;
	mat->element[0][1] = 2;
	mat->element[0][2] = 1;
	mat->element[1][0] = 0;
	mat->element[1][1] = 0;
	mat->element[1][2] = 0;
	mat->element[2][0] = -1;
	mat->element[2][1] = -2;
	mat->element[2][2] = -1;
}