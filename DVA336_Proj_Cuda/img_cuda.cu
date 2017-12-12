#include "opencv2\opencv.hpp"
#include "img_cuda.h"
#include "cuda_runtime.h"
#include "img_helper.h"
#include "device_launch_parameters.h"

#include <stdio.h>

using namespace cv;

#define BLOCKS 256
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

__global__ void kernel_sobel(int16_t * src, int16_t * dst, matrix mat, const int width, const int height) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int pixelValue;
	int pixelAcc = 0;
	const int noElements = width * height;

	while (index < noElements) {
		/* The if statement make sure that only pixels with eight neigbours are being affected */
		/*  NOT TOP ROW       && NOT BOTTOM ROW               && NOT FIRST COLUMN && NOT LAST COLUMN        */
		if (index > width - 1 && index < width*(height - 1) - 1 && index%width != 0 && index%width != width - 1) {
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
			pixelAcc = src[index];
		}
		dst[index] = pixelAcc;
		index += stride;
		pixelAcc = 0;
	}
}

/* Pixel pyth */
__global__ void kernel_pythagorean(int16_t *dst, int16_t *gx, int16_t *gy, const int width, const int height) {
	
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	const float compressionFactor = 255.0f / 1445.0f;

	int pixelGx, pixelGy;
	while (index < (width*height))
	{
		pixelGx = gx[index] * gx[index];
		pixelGy = gy[index] * gy[index];

		dst[index] = (int16_t)(sqrtf((float)pixelGx + (float)pixelGy) * compressionFactor); //Cast to float since CUDA sqrtf overload is float/double

		index += stride;
	}
}

__global__ void kernel_findMaxPixel(int16_t *src, const int width, const int height, int *maxPixel) {
	extern __shared__ int shared[];

	int tid = threadIdx.x;
	int gid = (blockDim.x * blockIdx.x) + tid;
	int stride = blockDim.x * gridDim.x;
	shared[tid] = -INT_MAX;  // 1
	const int elements = width*height;

	while (gid < elements)
	{
		if (gid < elements)
			shared[tid] = src[gid];
		__syncthreads();

		for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
		{
			if (tid < s && gid < elements)
				shared[tid] = max(shared[tid], shared[tid + s]);  // 2
			__syncthreads();
		}
		gid += stride;
	}
	
	// what to do now?
	// option 1: save block result and launch another kernel
	//if (tid == 0)
	//d_max[blockIdx.x] = shared[tid]; // 3
	// option 2: use atomics
	if (tid == 0)
	{
		atomicMax(maxPixel, shared[0]);
	}

}

__global__ void kernel_normalize(int16_t *src, const int width, const int height, int *maxPixel)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	const int elements = width * height;
	const float factor = 255.0f / (float)*maxPixel;

	while (index < elements)
	{
		src[index] = src[index] * factor;
		index += stride;
	}
}

void cuda_edge_detection(int16_t * src, Mat * image) {
	pixel * h_src_image;
	int16_t * h_dst_image;
	matrix matrix;
	pixel * d_src_image;
	int16_t * d_dst_image;
	int16_t * d_result_image;
	int16_t * d_sobelGx_image;
	int16_t * d_sobelGy_image;
	int * d_maxPixel;

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
	cudaMalloc((void**)&d_sobelGx_image, ext_elements * sizeof(int16_t));
	cudaMalloc((void**)&d_sobelGy_image, ext_elements * sizeof(int16_t));
	cudaMalloc((void**)&d_maxPixel, sizeof(int));

	/* Make grayscale*/
	cudaMemcpy(d_src_image, h_src_image, elements * sizeof(pixel), cudaMemcpyHostToDevice);
	kernel_grayscale <<<BLOCKS, THREADS>>>(d_src_image, d_dst_image, width, height);
	cudaDeviceSynchronize();

	/* Gaussian Blur */
	getGaussianKernel(&matrix);
	kernel_gaussian <<<BLOCKS, THREADS>>>(d_dst_image, d_result_image, matrix, width, height);
	cudaDeviceSynchronize();

	/* Multiplication with Gx */
	getGxKernel(&matrix);
	kernel_sobel <<<BLOCKS, THREADS>>>(d_result_image, d_sobelGx_image, matrix, width, height);
	cudaDeviceSynchronize();

	/* Multiplication with Gy */
	getGyKernel(&matrix);
	kernel_sobel <<<BLOCKS, THREADS>>>(d_result_image, d_sobelGy_image, matrix, width, height);
	cudaDeviceSynchronize();

	/* Pythagorean with Gx and Gy */
	kernel_pythagorean <<<BLOCKS, THREADS>>>(d_result_image, d_sobelGx_image, d_sobelGy_image, width, height);
	cudaDeviceSynchronize();

	/* Map values to max 255, allocate 4*THREADS bytes shared memory */
	kernel_findMaxPixel<<<BLOCKS,THREADS,4*THREADS>>>(d_result_image, width, height, d_maxPixel);
	cudaDeviceSynchronize();

	/* Map values to max 255, allocate 4*THREADS bytes shared memory */
	kernel_normalize <<<BLOCKS, THREADS>>>(d_result_image, width, height, d_maxPixel);
	cudaDeviceSynchronize();

	cudaMemcpy(src, d_result_image, elements * sizeof(int16_t), cudaMemcpyDeviceToHost);
	//cudaMemcpy(src, d_sobelGx_image, elements * sizeof(int16_t), cudaMemcpyDeviceToHost);

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