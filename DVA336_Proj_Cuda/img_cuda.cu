#include "opencv2\opencv.hpp"
#include "img_cuda.h"
#include "cuda_runtime.h"
#include "img_helper.h"
#include "device_launch_parameters.h"

#include <stdio.h>

using namespace cv;

__global__ void kernel_grayscale(pixel * src, int16_t * dst, const int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	if (index == 0) {
		printf("Stride : %d\n", stride);
	}
	while (index < n) {
		dst[index] = (int16_t)(src[index].b + src[index].r + src[index].g) / 3;
		index += stride;
	}
}

__global__ void kernel_gaussian() {

}

__global__ void kernel_sobel() {

}

__global__ void kernel_normalize() {

}

void cuda_edge_detection(int16_t * src, Mat * image) {
	pixel * h_src_image;
	int16_t * h_dst_image;
	pixel * d_src_image;
	int16_t * d_dst_image;
	int elements = (*image).cols * (*image).rows;

	h_src_image = (pixel *)malloc(elements * sizeof(pixel));
	h_dst_image = (int16_t *)malloc(elements * sizeof(int16_t));

	matToArray(image,h_src_image);

	cudaMalloc((void**)&d_src_image, elements * sizeof(pixel));
	cudaMalloc((void**)&d_dst_image, elements * sizeof(int16_t));
	cudaMemcpy(d_src_image, h_src_image, elements * sizeof(pixel), cudaMemcpyHostToDevice);
	kernel_grayscale << <64, 64 >> >(d_src_image, d_dst_image, elements);
	cudaDeviceSynchronize();
	cudaMemcpy(src, d_dst_image, elements * sizeof(int16_t), cudaMemcpyDeviceToHost);



	cudaFree(d_src_image);
	cudaFree(d_dst_image);
	free(h_src_image);
	free(h_dst_image);

}