#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include <vector>
#include <algorithm>

// This test does not compile

__global__ void mykernel(int N) {
  __device__ int mydevicefunc(int a)  {
    return 2*a; 
  }
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if ( i < N ) {
    printf("%d\n",mydevicefunc(i));
  }
}

int main(int argc, char** argv) {
  int N = 10;
  hipLaunchKernelGGL((mykernel),1,64,0,0,N);
  hipDeviceSynchronize();
}
