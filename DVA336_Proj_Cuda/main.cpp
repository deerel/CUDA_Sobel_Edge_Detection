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
	Mat src = imread("img\\input\\dresden_XL.jpg", CV_LOAD_IMAGE_COLOR);
	const int width = src.cols;
	const int height = src.rows;
	const int elements = width * height;

	pixel *pixel_array = (pixel*)calloc(elements, sizeof(pixel));
	matToArray(&src, pixel_array);

	Mat seq_image = src.clone();
	int16_t *seq_src = (int16_t *)calloc(elements, sizeof(int16_t));

	Mat cuda_image = src.clone();
	int16_t * cuda_src = (int16_t *)calloc(elements, sizeof(int16_t));

	chrono::high_resolution_clock::time_point start, stop;
	chrono::duration<float> execTime;

	float speedup;
	float cudatime, seqtime;
	

	/* CUDA */
	printf(".: CUDA :.\n");
	start = chrono::high_resolution_clock::now();

	cuda_edge_detection(cuda_src, pixel_array, width, height);

	stop = chrono::high_resolution_clock::now();
	execTime = chrono::duration_cast<chrono::duration<float>>(stop - start);
	printf("CUDA Exec time:       %f\n\n", execTime.count());
	speedup = execTime.count();
	cudatime = execTime.count();

	/* SEQ */
	printf(".: SEQ  :.\n");
	start = chrono::high_resolution_clock::now();

	seq_edge_detection(seq_src, pixel_array, width, height);

	stop = chrono::high_resolution_clock::now();
	execTime = chrono::duration_cast<chrono::duration<float>>(stop - start);
	printf("SEQ  Exec time:       %f\n\n", execTime.count());
	seqtime = execTime.count();

	printf("CUDA to SEQ speed up  %f\n", execTime.count() / speedup);

	compareImages(cuda_src, seq_src, elements);



    makeImage(seq_src, &seq_image);
	makeImage(cuda_src, &cuda_image);
	
	imshow("Seq edges", seq_image);

	imshow("Cuda edges", cuda_image);


	/* To output the image to disc */
	//vector<int> compression_params;
	//compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	//compression_params.push_back(9);
	//imwrite("img\\output\\output.png", seq_image, compression_params);
	
	waitKey();
	getchar();

	free(seq_src);
	free(cuda_src);
	free(pixel_array);

	return 0;
}
