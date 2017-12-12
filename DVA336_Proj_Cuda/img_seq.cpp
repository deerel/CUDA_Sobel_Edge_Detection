#include "img_seq.h"
#include "img_helper.h"

int16_t maxPixel;
int maxPos;

void convertToGrayscale(pixel *src, int16_t *dst, int len)
{
	for (int i = 0; i < len; i++)
	{
		dst[i] = (src[i].b + src[i].g + src[i].r) / 3;
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
		
		mapVal = mapToRange(sqrt(pixelGx + pixelGy), 1442, 255);
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

void seq_edge_detection(int16_t *src, Mat * image)
{
	int width = image->cols;
	int height = image->rows;
	int size = width*height;

	int16_t *srcMat = (int16_t *)calloc(size, sizeof(int16_t));
	int16_t *dstMat = (int16_t *)calloc(size, sizeof(int16_t));
	pixel *img = (pixel*)calloc(size, sizeof(pixel));

	int16_t *gxMat = (int16_t *)calloc(size, sizeof(int16_t));
	int16_t *gyMat = (int16_t *)calloc(size, sizeof(int16_t));
	matrix *kernel = (matrix*)calloc(1, sizeof(matrix));

	matToArray(image, img);

	convertToGrayscale(img, srcMat, size);

	getGaussianKernel(kernel);
	pixelMatMul(srcMat, dstMat, kernel, width, height, true);

	getGxKernel(kernel);
	pixelMatMul(dstMat, gxMat, kernel, width, height, false);

	getGyKernel(kernel);
	pixelMatMul(dstMat, gyMat, kernel, width, height, false);

	pixelPyth(src, gxMat, gyMat, width, height);

	normalize(src, width, height);
}

