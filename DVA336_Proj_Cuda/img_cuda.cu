#include "opencv2\opencv.hpp"
#include "img_cuda.h"
#include "cuda_runtime.h"
#include "img_helper.h"
#include "device_launch_parameters.h"

#include <stdio.h>

using namespace cv;

#define BLOCKS 512
#define THREADS 256

__global__ void kernel_grayscale(pixel * src, int16_t * dst, const int width, const int height) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	const int n = width * height;
	while (index < n) {
		dst[index] = (int16_t)(src[index].b + src[index].r + src[index].g) / 3;
		index += stride;
	}
}

// Extend this image with a 1 pixel border with value 0;
__global__ void kernel_gaussian(int16_t * src, int16_t * dst, matrix mat, const int width, const int height) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int pixelValue;
	int pixelAcc = 0;
	const int noElements = width * height;

	while (index < noElements) {
		/* The if statement make sure that only pixels with eight neigbours are being affected */
		/*  NOT TOP ROW       && NOT BOTTOM ROW               && NOT FIRST COLUMN && NOT LAST COLUMN        */
		if (index > width - 1 && index < width*(height - 1)-1 && index%width != 0 && index%width != width-1) {
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					int rowOffset = (i - 1)*width;
					int elementOffset = (j - 1);
					int pixel_index = index + rowOffset + elementOffset;

					pixelAcc += mat.element[i][j] * src[pixel_index];

				}
			}
		}
		else {
			//element is on the edge
			pixelAcc = src[index] * 16;
		}
		dst[index] = pixelAcc/16;
		index += stride;
		pixelAcc = 0;
	}

}

__global__ void kernel_sobel() {

}

__global__ void kernel_normalize() {

}

void cuda_edge_detection(int16_t * src, Mat * image) {
	pixel * h_src_image;
	int16_t * h_dst_image;
	matrix matrix;
	pixel * d_src_image;
	int16_t * d_dst_image;
	int16_t * d_result_image;

	const int width = image->cols;
	const int height = image->rows;

	const int elements = width * height;
	const int ext_elements = (width + 1) * (height + 1);

	h_src_image = (pixel *)malloc(elements * sizeof(pixel));
	h_dst_image = (int16_t *)malloc(elements * sizeof(int16_t));

	matToArray(image,h_src_image);

	cudaMalloc((void**)&d_src_image, elements * sizeof(pixel));
	cudaMalloc((void**)&d_dst_image, ext_elements * sizeof(int16_t));
	cudaMalloc((void**)&d_result_image, ext_elements * sizeof(int16_t));

	/* Make grayskale*/
	cudaMemcpy(d_src_image, h_src_image, elements * sizeof(pixel), cudaMemcpyHostToDevice);
	kernel_grayscale << <BLOCKS, THREADS >> >(d_src_image, d_dst_image, width, height);
	cudaDeviceSynchronize();

	/* Gaussian Blur */
	getGaussianKernel(&matrix);
	kernel_gaussian << <BLOCKS, THREADS >> >(d_dst_image, d_result_image, matrix, width, height);
	cudaDeviceSynchronize();

	cudaMemcpy(src, d_result_image, elements * sizeof(int16_t), cudaMemcpyDeviceToHost);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
	}

	cudaFree(d_src_image);
	cudaFree(d_dst_image);
	free(h_src_image);
	free(h_dst_image);

}