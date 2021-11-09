#include "gpufort.h"
#include "gpufort_arrays.h"

#include <assert.h>

__global__ void fill_int_array_1(
  gpufort::GpuArray1<int> arr
) {
  int i = -1+threadIdx.x + blockDim.x*blockIdx.x;
  if ( (i+1) < 10 ) {
    arr(i) = arr.linearized_index(i);
    //printf("%d\n",arr(i));
  }
}

__global__ void fill_int_array_2(
  gpufort::GpuArray2<int> arr
) {
  int i = -1+threadIdx.x + blockDim.x*blockIdx.x;
  int j = -2+threadIdx.y + blockDim.y*blockIdx.y;
  if ( (i+1) < 10 && (j+2) < 10 ) {
    arr(i,j) = arr.linearized_index(i,j);
  }
}

__global__ void fill_int_array_3(
  gpufort::GpuArray3<int> arr
) {
  int i = -1+threadIdx.x + blockDim.x*blockIdx.x;
  int j = -2+threadIdx.y + blockDim.y*blockIdx.y;
  int k = -3+threadIdx.z + blockDim.z*blockIdx.z;
  if ( (i+1) < 10 && (j+2) < 10 && (k+3) < 10 ) {
    arr(i,j,k) = arr.linearized_index(i,j,k);
  }
}

int main(int argc,char** argv) {
  gpufort::MappedArray1<int> int_array1;
  gpufort::MappedArray2<int> int_array2;
  gpufort::MappedArray3<int> int_array3;

  HIP_CHECK(int_array1.init(nullptr,nullptr, 10, -1, true)); // hostptr,devptr, count, lower bound, pinned
  HIP_CHECK(int_array2.init(nullptr,nullptr, 10,10, -1,-2, true));
  HIP_CHECK(int_array3.init(nullptr,nullptr, 10,10,10, -1,-2,-3, true));

  assert(  10==int_array1.data.num_elements);
  assert( 100==int_array2.data.num_elements);
  assert(1000==int_array3.data.num_elements);
  
  assert(  1==int_array1.data.index_offset);
  assert( 21==int_array2.data.index_offset);
  assert(321==int_array3.data.index_offset);
  
  assert(0==int_array1.data.linearized_index(-1));
  assert(0==int_array2.data.linearized_index(-1,-2));
  assert(0==int_array3.data.linearized_index(-1,-2,-3));

  assert(  9==int_array1.data.linearized_index(-1+10-1)); // upper bound
  assert( 99==int_array2.data.linearized_index(-1+10-1,-2+10-1));
  assert(999==int_array3.data.linearized_index(-1+10-1,-2+10-1,-3+10-1));

  hipLaunchKernelGGL(fill_int_array_1,dim3(1),dim3(10,1,1),0,nullptr,int_array1.data);
  hipLaunchKernelGGL(fill_int_array_2,dim3(1),dim3(10,10,1),0,nullptr,int_array2.data);
  hipLaunchKernelGGL(fill_int_array_3,dim3(1),dim3(10,10,10),0,nullptr,int_array3.data);

  HIP_CHECK(int_array1.copy_data_to_host()); 
  HIP_CHECK(int_array2.copy_data_to_host()); 
  HIP_CHECK(int_array3.copy_data_to_host()); 
  
  HIP_CHECK(hipDeviceSynchronize());

  for (int n = 0; n < int_array1.data.num_elements; n++ ) {
    //std::cout << int_array1.data.data_host[n] << std::endl; 
    assert(int_array1.data.data_host[n] == n);
  } 
  for (int n = 0; n < int_array2.data.num_elements; n++ ) {
    assert(int_array2.data.data_host[n] == n);
  } 
  for (int n = 0; n < int_array3.data.num_elements; n++ ) {
    assert(int_array3.data.data_host[n] == n);
  } 

  return 0;
}
