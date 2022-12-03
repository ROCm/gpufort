// SPDX-License-Identifier: MIT
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#include "gpufort_array.h"

__global__ void fill_int_array_1(
  gpufort::array1<int> arr
) {
  int i = -1+threadIdx.x + blockDim.x*blockIdx.x;
  if ( (i+1) < 10 ) {
    arr(i) = arr.c_idx(i);
    //printf("%d\n",arr(i));
  }
}

__global__ void fill_int_array_2(
  gpufort::array2<int> arr
) {
  int i = -1+threadIdx.x + blockDim.x*blockIdx.x;
  int j = -2+threadIdx.y + blockDim.y*blockIdx.y;
  if ( (i+1) < 10 && (j+2) < 10 ) {
    arr(i,j) = arr.c_idx(i,j);
  }
}

__global__ void fill_int_array_3(
  gpufort::array3<int> arr
) {
  int i = -1+threadIdx.x + blockDim.x*blockIdx.x;
  int j = -2+threadIdx.y + blockDim.y*blockIdx.y;
  int k = -3+threadIdx.z + blockDim.z*blockIdx.z;
  if ( (i+1) < 10 && (j+2) < 10 && (k+3) < 10 ) {
    arr(i,j,k) = arr.c_idx(i,j,k);
  }
}

extern "C" {
  void launch_fill_int_array_1(gpufort::array1<int> arr) {
    hipLaunchKernelGGL(fill_int_array_1,dim3(1),dim3(10,1,1),0,nullptr,arr);
  }
  void launch_fill_int_array_2(gpufort::array2<int> arr) {
    hipLaunchKernelGGL(fill_int_array_2,dim3(1),dim3(10,10,1),0,nullptr,arr);
  }
  void launch_fill_int_array_3(gpufort::array3<int> arr) {
    hipLaunchKernelGGL(fill_int_array_3,dim3(1),dim3(10,10,10),0,nullptr,arr);
  }
}