#include "kernel.h"

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

void addWithCuda(int *c, int *b, int *a, int size){
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaSetDevice(0);
	cudaMalloc((void**)&dev_c, size * sizeof(int));
	cudaMalloc((void**)&dev_a, size * sizeof(int));
	cudaMalloc((void**)&dev_b, size * sizeof(int));
	cudaMemcpy(dev_c, c, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	addKernel << <1, size >> >(dev_c, dev_a, dev_b);
	cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
}
