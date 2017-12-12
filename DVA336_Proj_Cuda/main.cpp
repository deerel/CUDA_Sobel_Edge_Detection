#include "opencv2\opencv.hpp"
#include <iostream>
#include <chrono>

#include "img_seq.h"
#include "img_cuda.h"
#include "img_helper.h"

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
	init_cuda();
	Mat src = imread("img\\input\\pontus.jpg", CV_LOAD_IMAGE_COLOR);
	
	Mat dst = src.clone();
	int16_t *dstMat = (int16_t *)calloc(src.cols * src.rows, sizeof(int16_t));

	Mat cuda_image = src.clone();
	int16_t * cuda_src = (int16_t *)calloc(src.cols * src.rows, sizeof(int16_t));

	chrono::high_resolution_clock::time_point start, stop;
	chrono::duration<float> execTime;
	float speedup;

	/* CUDA */
	printf(".: CUDA :.\n");
	start = chrono::high_resolution_clock::now();

	cuda_edge_detection(cuda_src, &cuda_image);

	stop = chrono::high_resolution_clock::now();
	execTime = chrono::duration_cast<chrono::duration<float>>(stop - start);
	printf("CUDA Exec time:       %f\n\n", execTime.count());
	speedup = execTime.count();

	/* SEQ */
	printf(".: SEQ  :.\n");
	start = chrono::high_resolution_clock::now();

	seq_edge_detection(dstMat, &dst);

	stop = chrono::high_resolution_clock::now();
	execTime = chrono::duration_cast<chrono::duration<float>>(stop - start);
	printf("SEQ  Exec time:       %f\n\n", execTime.count());

	printf("CUDA to SEQ speed up  %f\n", execTime.count() / speedup);

	makeImage(dstMat, &dst);
	makeImage(cuda_src, &cuda_image);
	
	imshow("Seq edges", dst);

	imshow("Cuda edges", cuda_image);

	/*vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	imwrite("img\\output\\output.png", dst, compression_params);*/
	waitKey();
	getchar();

	printf("Done\n");

	free(dstMat);
	free(cuda_src);
}
