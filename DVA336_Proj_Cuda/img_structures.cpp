#include "img_structures.h"

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