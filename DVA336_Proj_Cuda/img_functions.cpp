#include "img_functions.h"
int16_t maxPixel;
int maxPos;

void convertToGrayscale(Mat src, int16_t *dst)
{
	Vec3b original;
	int average;

	for (int r = 0; r < src.rows; r++)
	{
		for (int c = 0; c < src.cols; c++)
		{
			/* Get color value of pixel at index (r, c) */
			original = src.at<Vec3b>(r, c);

			/* Average the color values of pixel at index (r, c), ((B+G+R) / 3) */
			average = (original[0] + original[1] + original[2]) / 3;

			/* Set each color to the average */
			dst[c + r * src.cols] = (int16_t)average;
			
		}
	}
}

void makeImage(int16_t *src, Mat *dst)
{
	for (int r = 0; r < dst->rows; r++)
	{
		for (int c = 0; c < dst->cols; c++)
		{
			dst->at<Vec3b>(r, c)[0] = src[c + r * dst->cols];
			dst->at<Vec3b>(r, c)[1] = src[c + r * dst->cols];
			dst->at<Vec3b>(r, c)[2] = src[c + r * dst->cols];
		}
	}

}
/*
Neighbour detection: https://stackoverflow.com/questions/32502564/most-efficient-way-to-check-neighbouring-grid
*/
void pixelMatMul(int16_t *src, int16_t *dst, matrix *mat, int width, int height, bool divide)
{
	int arrayLen = width * height;
	int element, rowOffset, elementOffset, index, elementMod;

	for (int r = 0; r < height; r++)
	{
		for (int c = 0; c < width; c++)
		{
			int pixel = (int)src[c + r * width];
			int pixelAcc = 0;
			element = (c + r * width);
			elementMod = element % width;

			if (r > 0 && c > 0 && r < width - 1 && c < height - 1) {
				// element got all eight neighbours
				for (int i = 0; i < 3; i++)
				{
					rowOffset = (i - 1)*width;
					for (int j = 0; j < 3; j++)
					{
						elementOffset = (j - 1);
						index = element + rowOffset + elementOffset;

						pixelAcc += mat->element[i][j] * src[index];

						//if (i == 0 && r != 0)
						//{
						//	pixelAcc += mat->element[i][j] * src[(c + r * width) - width + (j - 1)];
						//}
						//else if (i == 1)
						//{
						//	pixelAcc += mat->element[i][j] * src[(c + r * width) + (j - 1)];
						//}
						//else if (i == 2 && r != height-1)
						//{
						//	pixelAcc += mat->element[i][j] * src[(c + r * width) + width + (j - 1)];
						//}
					}
				}
			}
			else {
				//element is on the edge
				divide ? pixelAcc = src[element] * 16 : pixelAcc = src[element];
			}

			divide ? pixelAcc /= 16 : pixelAcc;
			dst[c + r * width] = pixelAcc;
		}
	}



}

void pixelPyth(int16_t *dst, int16_t *gx, int16_t *gy, const int width, const int height)
{
	int pixelGx, pixelGy;
	uint16_t mapVal;
	maxPixel = 0;
	
	for (int i = 0; i < width*height; i++)
	{
		pixelGx = gx[i] * gx[i];
		pixelGy = gy[i] * gy[i];
		mapVal = mapToRange(sqrt(pixelGx + pixelGy), 1024, 255);
		dst[i] = mapVal;
		if (mapVal > maxPixel) {
			maxPixel = mapVal;
			maxPos = i;
		}
	}
}
void normalize(int16_t *src, const int width, const int height) {
	for (int i = 0; i < width*height; i++)
	{
		src[i] = mapToRange(src[i], maxPixel, 255);
	}
}

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
