#include <opencv2/opencv.hpp>  
#include <opencv2/gpu/gpu.hpp>    
#include <opencv2/gpu/stream_accessor.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::gpu;

void RGB2GrayWithCUDA()
{
	Mat src_image = imread("../testCUDAWithOpenCV/src/233.jpg");
	Mat dst_image;
	GpuMat gpu_src_image(src_image);
	GpuMat gpu_dst_image;
	cvtColor(gpu_src_image, gpu_dst_image, CV_RGB2GRAY);
	gpu_dst_image.download(dst_image);
	imshow("testWithCuda", dst_image);
	waitKey(0);
}

void swap_rb_caller(const PtrStepSz<uchar3>& src, PtrStep<uchar3> dst, cudaStream_t stream);

void swap_rb()
{
	Mat image = imread("../testCUDAWithOpenCV/src/233.jpg");
	imshow("src", image);
	GpuMat gpuMat, output;
	Stream& stream = Stream::Null();
	gpuMat.upload(image);
	CV_Assert(gpuMat.type() == CV_8UC3);
	output.create(gpuMat.size(), gpuMat.type());
	cudaStream_t s = StreamAccessor::getStream(stream);
	swap_rb_caller(gpuMat, output, s);
	output.download(image);

	imshow("gpu", image);
	waitKey(0);
}

int main(int argc, char* argv[])
{
	int iDevicesNum = getCudaEnabledDeviceCount();

	cout << iDevicesNum << endl;

	RGB2GrayWithCUDA();

	swap_rb();

	system("pause");

	return 0;
}