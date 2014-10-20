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

	addKernel << <1, size >> >(dev_c, dev_a, dev_b);
}
