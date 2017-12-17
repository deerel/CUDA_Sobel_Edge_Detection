#include <chrono>
#include "img_seq.h"
#include "img_helper.h"

#define SEQTIME 0

int16_t maxPixel;
int maxPos;

using namespace std;

void convertToGrayscale(pixel *src, int16_t *dst, const int len)
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

void gaussianBlur(int16_t * src, int16_t * dst, const int width, const int height)
{
	int element;
	int i, j;
	const int elements = width * height;

	for (int r = 0; r < height; r++)
	{
		for (int c = 0; c < width; c++)
		{
			int pixel = (int)src[c + r * width];
			int pixelAcc = 0;
			element = (c + r * width);

			if (r > 0 && c > 0 && r < height - 1 && c < width - 1) {
				pixelAcc += src[element - width - 1];			// 1/16
				pixelAcc += src[element - width] * 2;			// 2/16
				pixelAcc += src[element - width + 1];			// 1/16
				pixelAcc += src[element - 1] * 2;				// 2/16
				pixelAcc += src[element] * 4;					// 4/16
				pixelAcc += src[element + 1] * 2;				// 2/16
				pixelAcc += src[element + width - 1];			// 1/16
				pixelAcc += src[element + width] * 2;			// 2/16
				pixelAcc += src[element + width + 1];			// 1/16
				dst[element] = pixelAcc / 16;
			}
			else {
				//element is on the edge
				dst[element] = src[element];
			}
		}
	}

}

void sobel_gx(int16_t * src, int16_t *dst, const int width, const int height)
{
	int element;
	for (int r = 0; r < height; r++)
	{
		for (int c = 0; c < width; c++)
		{
			int pixelAcc = 0;
			element = (c + r * width);

			if (r > 0 && c > 0 && r < height - 1 && c < width - 1) {
				pixelAcc += src[element - width - 1];
				pixelAcc += src[element - width + 1] * -1;
				pixelAcc += src[element - 1] * 2;
				pixelAcc += src[element + 1] * -2;
				pixelAcc += src[element + width - 1];
				pixelAcc += src[element + width + 1] * -1;
			}
			dst[c + r * width] = pixelAcc;
		}
	}
}


void sobel_gy(int16_t * src, int16_t *dst, const int width, const int height)
{
	int element;
	for (int r = 0; r < height; r++)
	{
		for (int c = 0; c < width; c++)
		{
			int pixelAcc = 0;
			element = (c + r * width);

			if (r > 0 && c > 0 && r < height - 1 && c < width - 1) {
				pixelAcc += src[element - width - 1];
				pixelAcc += src[element - width] * 2;
				pixelAcc += src[element - width + 1];
				pixelAcc += src[element + width - 1] * -1;
				pixelAcc += src[element + width] * -2;
				pixelAcc += src[element + width + 1] * -1;
			}
			dst[c + r * width] = pixelAcc;
		}
	}
}

void pixelPyth(int16_t *dst, int16_t * gx, int16_t * gy, const int width, const int height)
{
	int pixelGx, pixelGy;
	int pixelGx2, pixelGy2;
	uint16_t mapVal;
	uint16_t mapVal2;
	uint16_t max;
	int index;
	maxPixel = 0;
	const float compressFactor = 255.0f / 1445.0f;
	const int elements = width * height;
	
	/* Since it is a 2D image we can unroll with 2 to gain performance */
	for (int i = 0; i < elements; i+=2)
	{
		pixelGx = gx[i] * gx[i];
		pixelGy = gy[i] * gy[i];
		pixelGx2 = gx[i+1] * gx[i + 1];
		pixelGy2 = gy[i + 1] * gy[i + 1];
		
		mapVal = (int)(sqrtf((float)pixelGx + (float)pixelGy) * compressFactor);
		mapVal2 = (int)(sqrtf((float)pixelGx2 + (float)pixelGy2) * compressFactor);
		dst[i] = mapVal;
		dst[i+1] = mapVal2;

		if (mapVal > mapVal2) {
			max = mapVal;
			index = i;
		}
		else {
			max = mapVal2;
			index = i+1;
		}
		if (max > maxPixel) {
			maxPixel = max;
			maxPos = index;
		}
	}
}

void normalize(int16_t *src, const int width, const int height) {

	const int elements = width * height;
	const float factor = 255.0f / (float)maxPixel;

	/* Since it is a 2D image we can unroll with 2 to gain performance */
	for (int i = 0; i <elements; i+=2)
	{
		src[i] = src[i] * factor;	
		src[i+1] = src[i+1] * factor;
	}
}

void seq_edge_detection(int16_t *src, pixel * pixel_array, const int width, const int height)
{
	const int size = width*height;

#if SEQTIME > 0
	chrono::high_resolution_clock::time_point start, stop;
	chrono::duration<float> execTime;
	start = chrono::high_resolution_clock::now();
#endif
	int16_t *int16_array_1 = (int16_t *)calloc(size, sizeof(int16_t));
	int16_t *int16_array_2 = (int16_t *)calloc(size, sizeof(int16_t));

	int16_t *gxMat = (int16_t *)calloc(size, sizeof(int16_t));
	int16_t *gyMat = (int16_t *)calloc(size, sizeof(int16_t));
	matrix *kernel = (matrix*)calloc(1, sizeof(matrix));
#if SEQTIME > 0
	stop = chrono::high_resolution_clock::now();
	execTime = chrono::duration_cast<chrono::duration<float>>(stop - start);
	printf("Malloc time:          %f\n", execTime.count());
#endif

#if SEQTIME > 0
	start = chrono::high_resolution_clock::now();
#endif
	convertToGrayscale(pixel_array, int16_array_1, size);
#if SEQTIME > 0
	stop = chrono::high_resolution_clock::now();
	execTime = chrono::duration_cast<chrono::duration<float>>(stop - start);
	printf("Grayscale time:       %f\n", execTime.count());
#endif

#if SEQTIME > 0
	start = chrono::high_resolution_clock::now();
#endif
	gaussianBlur(int16_array_1, int16_array_2, width, height);
#if SEQTIME > 0
	stop = chrono::high_resolution_clock::now();
	execTime = chrono::duration_cast<chrono::duration<float>>(stop - start);
	printf("Gaussian time:        %f\n", execTime.count());
#endif

#if SEQTIME > 0
	start = chrono::high_resolution_clock::now();
#endif
	sobel_gx(int16_array_2, gxMat, width, height);
#if SEQTIME > 0
	stop = chrono::high_resolution_clock::now();
	execTime = chrono::duration_cast<chrono::duration<float>>(stop - start);
	printf("Gx time:              %f\n", execTime.count());
#endif

#if SEQTIME > 0
	start = chrono::high_resolution_clock::now();
#endif
	sobel_gy(int16_array_2, gyMat, width, height);
#if SEQTIME > 0
	stop = chrono::high_resolution_clock::now();
	execTime = chrono::duration_cast<chrono::duration<float>>(stop - start);
	printf("Gy time:              %f\n", execTime.count());
#endif

#if SEQTIME > 0
	start = chrono::high_resolution_clock::now();
#endif
	pixelPyth(src, gxMat, gyMat, width, height);
#if SEQTIME > 0
	stop = chrono::high_resolution_clock::now();
	execTime = chrono::duration_cast<chrono::duration<float>>(stop - start);
	printf("Pyth time:            %f\n", execTime.count());
#endif

#if SEQTIME > 0
	start = chrono::high_resolution_clock::now();
#endif
	normalize(src, width, height);
#if SEQTIME > 0
	stop = chrono::high_resolution_clock::now();
	execTime = chrono::duration_cast<chrono::duration<float>>(stop - start);
	printf("Normalize time:       %f\n", execTime.count());
#endif

#if SEQTIME > 0
	start = chrono::high_resolution_clock::now();
#endif
	free(int16_array_1);
	free(int16_array_2);
	free(gxMat);
	free(gyMat);
	free(kernel);
#if SEQTIME > 0
	stop = chrono::high_resolution_clock::now();
	execTime = chrono::duration_cast<chrono::duration<float>>(stop - start);
	printf("Free time:            %f\n", execTime.count());
#endif

}



