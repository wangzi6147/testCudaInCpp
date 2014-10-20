#include "kernel.h"
#include <iostream>

using namespace::std;

int main(){
	const int arraySize = 5;
	int a[arraySize] = { 1, 2, 3, 4, 5 };
	int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	addWithCuda(c,a,b,arraySize);

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);
	system("pause");
}
