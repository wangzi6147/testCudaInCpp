#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void addWithCuda(int *c , int *b, int *a , int size);