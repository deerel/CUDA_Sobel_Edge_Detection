#include "img_helper.h"

using namespace cv;

int16_t mapToRange(int16_t x, int16_t srcMax, uint16_t dstMax)
{
	float mapVal = ((float)x / (float)srcMax) * (float)dstMax;
	return (int16_t)mapVal;
}

void matToArray(Mat src, pixel * dst)
{
	for (int r = 0; r < src.rows; r++)
	{
		for (int c = 0; c < src.cols; c++)
		{
			dst[c + r * src.cols].b = src.at<Vec3b>(r, c)[0];
			dst[c + r * src.cols].g = src.at<Vec3b>(r, c)[1];
			dst[c + r * src.cols].r = src.at<Vec3b>(r, c)[2];
		}
	}
}
