#include "opencv2\opencv.hpp"
#include <iostream>

#include "img_functions.h"

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
	Mat src = imread("img\\input\\rickard.jpg", CV_LOAD_IMAGE_COLOR);
	Mat dst = src.clone();
	int16_t *srcMat = (int16_t *)calloc(src.cols * src.rows, sizeof(int16_t));
	int16_t *dstMat = (int16_t *)calloc(src.cols * src.rows, sizeof(int16_t));

	int16_t *gxMat = (int16_t *)calloc(src.cols * src.rows, sizeof(int16_t));
	int16_t *gyMat = (int16_t *)calloc(src.cols * src.rows, sizeof(int16_t));
	matrix *kernel = (matrix*)calloc(1, sizeof(matrix));

	convertToGrayscale(src, srcMat);

	getGaussianKernel(kernel);
	printf("Blur\n");
	pixelMatMul(srcMat, dstMat, kernel, src.cols, src.rows, true);

	getGxKernel(kernel);
	printf("GX\n");
	pixelMatMul(dstMat, gxMat, kernel, src.cols, src.rows, false);

	getGyKernel(kernel);
	printf("GY\n");
	pixelMatMul(dstMat, gyMat, kernel, src.cols, src.rows, false);

	pixelPyth(dstMat, gxMat, gyMat, src.cols, src.rows);

	normalize(dstMat, src.cols, src.rows);

	makeImage(dstMat, &dst);
	
	imshow("Original", src);
	imshow("Grayscale", dst);

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	imwrite("img\\output\\output.png", dst, compression_params);
	waitKey();

	free(srcMat);
	free(dstMat);
	free(kernel);
}
