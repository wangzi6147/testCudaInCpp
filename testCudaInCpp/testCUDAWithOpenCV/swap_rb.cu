#include <opencv2/core/cuda_devptrs.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace cv;
using namespace cv::gpu;
//自定义内核函数
__global__ void swap_rb_kernel(const PtrStepSz<uchar3> src, PtrStep<uchar3> dst)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < src.cols && y < src.rows)
	{
		uchar3 v = src(y, x);
		dst(y, x) = make_uchar3(v.z, v.y, v.x);
	}
}

void swap_rb_caller(const PtrStepSz<uchar3>& src, PtrStep<uchar3> dst, cudaStream_t stream)
{
	dim3 block(32, 8);
	dim3 grid((src.cols + block.x - 1) / block.x, (src.rows + block.y - 1) / block.y);

	swap_rb_kernel << <grid, block, 0, stream >> >(src, dst);
	if (stream == 0)
		cudaDeviceSynchronize();
}