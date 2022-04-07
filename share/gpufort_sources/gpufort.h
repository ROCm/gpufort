// This file was generated from a template via gpufort --gpufort-create-headers
#ifndef _GPUFORT_H_
#define _GPUFORT_H_
#include "hip/hip_complex.h"
#include "hip/math_functions.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <utility>
#include <algorithm>
#include <vector>
#define HIP_CHECK(condition)         \
  {                                  \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
  }

// global thread indices for various dimensions
#define __gidx(idx) (threadIdx.idx + blockIdx.idx * blockDim.idx)
#define __gidx1 __gidx(x)
#define __gidx2 (__gidx(x) + gridDim.x*blockDim.x*__gidx(y))
#define __gidx3 (__gidx(x) + gridDim.x*blockDim.x*__gidx(y) + gridDim.x*blockDim.x*gridDim.y*blockDim.y*__gidx(z))
#define __total_threads(grid,block) ( (grid).x*(grid).y*(grid).z * (block).x*(block).y*(block).z )
#define divideAndRoundUp(x, y) ((x) / (y) + ((x) % (y) != 0))

#define GPUFORT_PRINT_ARGS(...) gpufort_show_args(std::cout, #__VA_ARGS__, __VA_ARGS__)
namespace {
  template<typename H1> std::ostream& gpufort_show_args(std::ostream& out, const char* label, H1&& value) {
    return out << label << "=" << std::forward<H1>(value) << '\n';
  }
  
  template<typename H1, typename ...T> std::ostream& gpufort_show_args(std::ostream& out, const char* label, H1&& value, T&&... rest) {
    const char* pcomma = strchr(label, ',');
    return gpufort_show_args(out.write(label, pcomma - label) << "=" << std::forward<H1>(value) << ',',
          pcomma + 1,
          std::forward<T>(rest)...);
  }
}
#define GPUFORT_PRINT_ARRAY1(prefix,print_values,print_norms,A,n1,lb1) gpufort_print_array1(std::cout, prefix, print_values, print_norms, #A, A, n1,lb1)
#define GPUFORT_PRINT_ARRAY2(prefix,print_values,print_norms,A,n1,n2,lb1,lb2) gpufort_print_array2(std::cout, prefix, print_values, print_norms, #A, A, n1,n2,lb1,lb2)
#define GPUFORT_PRINT_ARRAY3(prefix,print_values,print_norms,A,n1,n2,n3,lb1,lb2,lb3) gpufort_print_array3(std::cout, prefix, print_values, print_norms, #A, A, n1,n2,n3,lb1,lb2,lb3)
#define GPUFORT_PRINT_ARRAY4(prefix,print_values,print_norms,A,n1,n2,n3,n4,lb1,lb2,lb3,lb4) gpufort_print_array4(std::cout, prefix, print_values, print_norms, #A, A, n1,n2,n3,n4,lb1,lb2,lb3,lb4)
#define GPUFORT_PRINT_ARRAY5(prefix,print_values,print_norms,A,n1,n2,n3,n4,n5,lb1,lb2,lb3,lb4,lb5) gpufort_print_array5(std::cout, prefix, print_values, print_norms, #A, A, n1,n2,n3,n4,n5,lb1,lb2,lb3,lb4,lb5)
#define GPUFORT_PRINT_ARRAY6(prefix,print_values,print_norms,A,n1,n2,n3,n4,n5,n6,lb1,lb2,lb3,lb4,lb5,lb6) gpufort_print_array6(std::cout, prefix, print_values, print_norms, #A, A, n1,n2,n3,n4,n5,n6,lb1,lb2,lb3,lb4,lb5,lb6)
#define GPUFORT_PRINT_ARRAY7(prefix,print_values,print_norms,A,n1,n2,n3,n4,n5,n6,n7,lb1,lb2,lb3,lb4,lb5,lb6,lb7) gpufort_print_array7(std::cout, prefix, print_values, print_norms, #A, A, n1,n2,n3,n4,n5,n6,n7,lb1,lb2,lb3,lb4,lb5,lb6,lb7)
namespace {
  template<typename T>
  void gpufort_print_array1(std::ostream& out, const char* prefix, const bool print_values, const bool print_norms, const char* label, T A[], int n1,int lb1) {
    int n =n1;
    std::vector<T> A_h(n);
    out << prefix << label << ":" << "\n";
    HIP_CHECK(hipMemcpy(A_h.data(), A, n*sizeof(T), hipMemcpyDeviceToHost)); 
    T min = +std::numeric_limits<T>::max();
    T max = -std::numeric_limits<T>::max();
    T sum = 0;
    T l1  = 0;
    T l2  = 0;
    for ( int i1 = 0; i1 < n1; i1++ ) {
      T value = A_h[i1];
      if ( print_norms ) {
        min  = std::min(value,min);
        max  = std::max(value,max);
        sum += value;
        l1  += std::abs(value);
        l2  += value*value;
      }
      if ( print_values ) {
        out << prefix << label << "(" << (lb1+i1) <<  ") = " << std::setprecision(6) << value << "\n";
      }
    } // for loops
    if ( print_norms ) {
      out << prefix << label << ":" << "min=" << min << "\n";
      out << prefix << label << ":" << "max=" << max << "\n";
      out << prefix << label << ":" << "sum=" << sum << "\n";
      out << prefix << label << ":" << "l1="  << l1  << "\n";
      out << prefix << label << ":" << "l2="  << std::sqrt(l2) << "\n";
    }
  }
  template<typename T>
  void gpufort_print_array2(std::ostream& out, const char* prefix, const bool print_values, const bool print_norms, const char* label, T A[], int n1,int n2,int lb1,int lb2) {
    int n =n1*n2;
    std::vector<T> A_h(n);
    out << prefix << label << ":" << "\n";
    HIP_CHECK(hipMemcpy(A_h.data(), A, n*sizeof(T), hipMemcpyDeviceToHost)); 
    T min = +std::numeric_limits<T>::max();
    T max = -std::numeric_limits<T>::max();
    T sum = 0;
    T l1  = 0;
    T l2  = 0;
    for ( int i2 = 0; i2 < n2; i2++ ) {
    for ( int i1 = 0; i1 < n1; i1++ ) {
      T value = A_h[n1*i2+i1];
      if ( print_norms ) {
        min  = std::min(value,min);
        max  = std::max(value,max);
        sum += value;
        l1  += std::abs(value);
        l2  += value*value;
      }
      if ( print_values ) {
        out << prefix << label << "(" << (lb1+i1) << "," << (lb2+i2) <<  ") = " << std::setprecision(6) << value << "\n";
      }
    }} // for loops
    if ( print_norms ) {
      out << prefix << label << ":" << "min=" << min << "\n";
      out << prefix << label << ":" << "max=" << max << "\n";
      out << prefix << label << ":" << "sum=" << sum << "\n";
      out << prefix << label << ":" << "l1="  << l1  << "\n";
      out << prefix << label << ":" << "l2="  << std::sqrt(l2) << "\n";
    }
  }
  template<typename T>
  void gpufort_print_array3(std::ostream& out, const char* prefix, const bool print_values, const bool print_norms, const char* label, T A[], int n1,int n2,int n3,int lb1,int lb2,int lb3) {
    int n =n1*n2*n3;
    std::vector<T> A_h(n);
    out << prefix << label << ":" << "\n";
    HIP_CHECK(hipMemcpy(A_h.data(), A, n*sizeof(T), hipMemcpyDeviceToHost)); 
    T min = +std::numeric_limits<T>::max();
    T max = -std::numeric_limits<T>::max();
    T sum = 0;
    T l1  = 0;
    T l2  = 0;
    for ( int i3 = 0; i3 < n3; i3++ ) {
    for ( int i2 = 0; i2 < n2; i2++ ) {
    for ( int i1 = 0; i1 < n1; i1++ ) {
      T value = A_h[n1*n2*i3+n1*i2+i1];
      if ( print_norms ) {
        min  = std::min(value,min);
        max  = std::max(value,max);
        sum += value;
        l1  += std::abs(value);
        l2  += value*value;
      }
      if ( print_values ) {
        out << prefix << label << "(" << (lb1+i1) << "," << (lb2+i2) << "," << (lb3+i3) <<  ") = " << std::setprecision(6) << value << "\n";
      }
    }}} // for loops
    if ( print_norms ) {
      out << prefix << label << ":" << "min=" << min << "\n";
      out << prefix << label << ":" << "max=" << max << "\n";
      out << prefix << label << ":" << "sum=" << sum << "\n";
      out << prefix << label << ":" << "l1="  << l1  << "\n";
      out << prefix << label << ":" << "l2="  << std::sqrt(l2) << "\n";
    }
  }
  template<typename T>
  void gpufort_print_array4(std::ostream& out, const char* prefix, const bool print_values, const bool print_norms, const char* label, T A[], int n1,int n2,int n3,int n4,int lb1,int lb2,int lb3,int lb4) {
    int n =n1*n2*n3*n4;
    std::vector<T> A_h(n);
    out << prefix << label << ":" << "\n";
    HIP_CHECK(hipMemcpy(A_h.data(), A, n*sizeof(T), hipMemcpyDeviceToHost)); 
    T min = +std::numeric_limits<T>::max();
    T max = -std::numeric_limits<T>::max();
    T sum = 0;
    T l1  = 0;
    T l2  = 0;
    for ( int i4 = 0; i4 < n4; i4++ ) {
    for ( int i3 = 0; i3 < n3; i3++ ) {
    for ( int i2 = 0; i2 < n2; i2++ ) {
    for ( int i1 = 0; i1 < n1; i1++ ) {
      T value = A_h[n1*n2*n3*i4+n1*n2*i3+n1*i2+i1];
      if ( print_norms ) {
        min  = std::min(value,min);
        max  = std::max(value,max);
        sum += value;
        l1  += std::abs(value);
        l2  += value*value;
      }
      if ( print_values ) {
        out << prefix << label << "(" << (lb1+i1) << "," << (lb2+i2) << "," << (lb3+i3) << "," << (lb4+i4) <<  ") = " << std::setprecision(6) << value << "\n";
      }
    }}}} // for loops
    if ( print_norms ) {
      out << prefix << label << ":" << "min=" << min << "\n";
      out << prefix << label << ":" << "max=" << max << "\n";
      out << prefix << label << ":" << "sum=" << sum << "\n";
      out << prefix << label << ":" << "l1="  << l1  << "\n";
      out << prefix << label << ":" << "l2="  << std::sqrt(l2) << "\n";
    }
  }
  template<typename T>
  void gpufort_print_array5(std::ostream& out, const char* prefix, const bool print_values, const bool print_norms, const char* label, T A[], int n1,int n2,int n3,int n4,int n5,int lb1,int lb2,int lb3,int lb4,int lb5) {
    int n =n1*n2*n3*n4*n5;
    std::vector<T> A_h(n);
    out << prefix << label << ":" << "\n";
    HIP_CHECK(hipMemcpy(A_h.data(), A, n*sizeof(T), hipMemcpyDeviceToHost)); 
    T min = +std::numeric_limits<T>::max();
    T max = -std::numeric_limits<T>::max();
    T sum = 0;
    T l1  = 0;
    T l2  = 0;
    for ( int i5 = 0; i5 < n5; i5++ ) {
    for ( int i4 = 0; i4 < n4; i4++ ) {
    for ( int i3 = 0; i3 < n3; i3++ ) {
    for ( int i2 = 0; i2 < n2; i2++ ) {
    for ( int i1 = 0; i1 < n1; i1++ ) {
      T value = A_h[n1*n2*n3*n4*i5+n1*n2*n3*i4+n1*n2*i3+n1*i2+i1];
      if ( print_norms ) {
        min  = std::min(value,min);
        max  = std::max(value,max);
        sum += value;
        l1  += std::abs(value);
        l2  += value*value;
      }
      if ( print_values ) {
        out << prefix << label << "(" << (lb1+i1) << "," << (lb2+i2) << "," << (lb3+i3) << "," << (lb4+i4) << "," << (lb5+i5) <<  ") = " << std::setprecision(6) << value << "\n";
      }
    }}}}} // for loops
    if ( print_norms ) {
      out << prefix << label << ":" << "min=" << min << "\n";
      out << prefix << label << ":" << "max=" << max << "\n";
      out << prefix << label << ":" << "sum=" << sum << "\n";
      out << prefix << label << ":" << "l1="  << l1  << "\n";
      out << prefix << label << ":" << "l2="  << std::sqrt(l2) << "\n";
    }
  }
  template<typename T>
  void gpufort_print_array6(std::ostream& out, const char* prefix, const bool print_values, const bool print_norms, const char* label, T A[], int n1,int n2,int n3,int n4,int n5,int n6,int lb1,int lb2,int lb3,int lb4,int lb5,int lb6) {
    int n =n1*n2*n3*n4*n5*n6;
    std::vector<T> A_h(n);
    out << prefix << label << ":" << "\n";
    HIP_CHECK(hipMemcpy(A_h.data(), A, n*sizeof(T), hipMemcpyDeviceToHost)); 
    T min = +std::numeric_limits<T>::max();
    T max = -std::numeric_limits<T>::max();
    T sum = 0;
    T l1  = 0;
    T l2  = 0;
    for ( int i6 = 0; i6 < n6; i6++ ) {
    for ( int i5 = 0; i5 < n5; i5++ ) {
    for ( int i4 = 0; i4 < n4; i4++ ) {
    for ( int i3 = 0; i3 < n3; i3++ ) {
    for ( int i2 = 0; i2 < n2; i2++ ) {
    for ( int i1 = 0; i1 < n1; i1++ ) {
      T value = A_h[n1*n2*n3*n4*n5*i6+n1*n2*n3*n4*i5+n1*n2*n3*i4+n1*n2*i3+n1*i2+i1];
      if ( print_norms ) {
        min  = std::min(value,min);
        max  = std::max(value,max);
        sum += value;
        l1  += std::abs(value);
        l2  += value*value;
      }
      if ( print_values ) {
        out << prefix << label << "(" << (lb1+i1) << "," << (lb2+i2) << "," << (lb3+i3) << "," << (lb4+i4) << "," << (lb5+i5) << "," << (lb6+i6) <<  ") = " << std::setprecision(6) << value << "\n";
      }
    }}}}}} // for loops
    if ( print_norms ) {
      out << prefix << label << ":" << "min=" << min << "\n";
      out << prefix << label << ":" << "max=" << max << "\n";
      out << prefix << label << ":" << "sum=" << sum << "\n";
      out << prefix << label << ":" << "l1="  << l1  << "\n";
      out << prefix << label << ":" << "l2="  << std::sqrt(l2) << "\n";
    }
  }
  template<typename T>
  void gpufort_print_array7(std::ostream& out, const char* prefix, const bool print_values, const bool print_norms, const char* label, T A[], int n1,int n2,int n3,int n4,int n5,int n6,int n7,int lb1,int lb2,int lb3,int lb4,int lb5,int lb6,int lb7) {
    int n =n1*n2*n3*n4*n5*n6*n7;
    std::vector<T> A_h(n);
    out << prefix << label << ":" << "\n";
    HIP_CHECK(hipMemcpy(A_h.data(), A, n*sizeof(T), hipMemcpyDeviceToHost)); 
    T min = +std::numeric_limits<T>::max();
    T max = -std::numeric_limits<T>::max();
    T sum = 0;
    T l1  = 0;
    T l2  = 0;
    for ( int i7 = 0; i7 < n7; i7++ ) {
    for ( int i6 = 0; i6 < n6; i6++ ) {
    for ( int i5 = 0; i5 < n5; i5++ ) {
    for ( int i4 = 0; i4 < n4; i4++ ) {
    for ( int i3 = 0; i3 < n3; i3++ ) {
    for ( int i2 = 0; i2 < n2; i2++ ) {
    for ( int i1 = 0; i1 < n1; i1++ ) {
      T value = A_h[n1*n2*n3*n4*n5*n6*i7+n1*n2*n3*n4*n5*i6+n1*n2*n3*n4*i5+n1*n2*n3*i4+n1*n2*i3+n1*i2+i1];
      if ( print_norms ) {
        min  = std::min(value,min);
        max  = std::max(value,max);
        sum += value;
        l1  += std::abs(value);
        l2  += value*value;
      }
      if ( print_values ) {
        out << prefix << label << "(" << (lb1+i1) << "," << (lb2+i2) << "," << (lb3+i3) << "," << (lb4+i4) << "," << (lb5+i5) << "," << (lb6+i6) << "," << (lb7+i7) <<  ") = " << std::setprecision(6) << value << "\n";
      }
    }}}}}}} // for loops
    if ( print_norms ) {
      out << prefix << label << ":" << "min=" << min << "\n";
      out << prefix << label << ":" << "max=" << max << "\n";
      out << prefix << label << ":" << "sum=" << sum << "\n";
      out << prefix << label << ":" << "l1="  << l1  << "\n";
      out << prefix << label << ":" << "l2="  << std::sqrt(l2) << "\n";
    }
  }  
  /**
   * Checks if `idx` has reached the end of the loop iteration
   * range yet. 
   *
   * \note Takes only the sign of `step` into account, not its value.
   *
   * \param[in] idx loop index
   * \param[in] begin begin of the loop iteration range
   * \param[in] end end of the loop iteration range
   * \param[in] step step size of the loop iteration range
   */
  __device__ __forceinline__ bool loop_cond(int idx,int begin,int end,int step) {
    return (step>0) ? ( idx <= end ) : ( -idx <= -end );     
  }
  
  /** 
   * Overloaded variant that deduces the sign of a unit step
   * based on inputs `begin` and `end`.
   */ 
  __device__ __forceinline__ bool loop_cond(int idx,int begin,int end) {
    int step = ( begin <= end ) ? 1 : -1; 
    return loop_cond(idx,begin,end,step);
  }
  
  /**
   * Given the index for iterating a collapsed loop nest
   * and the number of iterations of that collapsed loop nest,
   * this function returns the index of the outermost loop
   * of the original (uncollapsed) loop nest.
   *
   * \return index for iterating the original outermost loop.
   *
   * \note Side effects: Argument `index`
   *       is decremented according to the number of iterations
   *       of the outermost loop. It can then be used to retrieve
   *       the index of the next inner loop, and so on.
   *       Argument `problem_size` is divided by the number of iterations of the outermost loop.
   *       It can then also be passed directly to the next call of `outermost_index`.
   * \param[inout] index index for iterating collapsed loop nest. This
   *                     is not the index of the outermost loop!
   * \param[inout] problem_size Denominator for retrieving outermost loop index. Must be chosen
   *                           equal to the total number of iterations of the collapsed loop nest 
   *                           before the first call of `outermost_index`.
   * \param[in] begin begin of the outermost loop iteration range
   * \param[in] end end of the outermost loop iteration range
   * \param[in] step step size of the outermost loop iteration range
   */
  __host__ __device__ __forceinline__ int outermost_index(
    int& index,
    int& problem_size,
    const int begin, const int end, const int step
  ) {
    const int size = (abs(end - begin) + 1)/abs(step);
    problem_size /= size;
    const int idx = index / problem_size; // rounds down
    index -= idx*problem_size;
    return (begin + step*idx);
  }
 
  /** 
   * Overloaded variant that deduces the sign of a unit step
   * based on inputs `begin` and `end`.
   */ 
  __host__ __device__ __forceinline__ int outermost_index(
    int& index,
    int& problem_size,
    const int begin, const int end
  ) {
    int step = ( begin <= end ) ? 1 : -1; 
    return outermost_index(index,problem_size,begin,end,step);
  }

  // type conversions (complex make routines already defined via "hip/hip_complex.h")
  // make float
  __device__ __forceinline__ float make_float(const short int a) {
    return static_cast<float>(a);
  }
  __device__ __forceinline__ float make_float(const unsigned short int a) {
    return static_cast<float>(a);
  }
  __device__ __forceinline__ float make_float(const unsigned int a) {
    return static_cast<float>(a);
  }
  __device__ __forceinline__ float make_float(const int a) {
    return static_cast<float>(a);
  }
  __device__ __forceinline__ float make_float(const long int a) {
    return static_cast<float>(a);
  }
  __device__ __forceinline__ float make_float(const unsigned long int a) {
    return static_cast<float>(a);
  }
  __device__ __forceinline__ float make_float(const long long int a) {
    return static_cast<float>(a);
  }
  __device__ __forceinline__ float make_float(const unsigned long long int a) {
    return static_cast<float>(a);
  }
  __device__ __forceinline__ float make_float(const signed char a) {
    return static_cast<float>(a);
  }
  __device__ __forceinline__ float make_float(const unsigned char a) {
    return static_cast<float>(a);
  }
  __device__ __forceinline__ float make_float(const float a) {
    return static_cast<float>(a);
  }
  __device__ __forceinline__ float make_float(const double a) {
    return static_cast<float>(a);
  }
  __device__ __forceinline__ float make_float(const long double a) {
    return static_cast<float>(a);
  }
  __device__ __forceinline__ float make_float(const hipFloatComplex a) {
    return static_cast<float>(a.x);
  }
  __device__ __forceinline__ float make_float(const hipDoubleComplex a) {
    return static_cast<float>(a.x);
  }
  // make double
  __device__ __forceinline__ double make_double(const short int a) {
    return static_cast<double>(a);
  }
  __device__ __forceinline__ double make_double(const unsigned short int a) {
    return static_cast<double>(a);
  }
  __device__ __forceinline__ double make_double(const unsigned int a) {
    return static_cast<double>(a);
  }
  __device__ __forceinline__ double make_double(const int a) {
    return static_cast<double>(a);
  }
  __device__ __forceinline__ double make_double(const long int a) {
    return static_cast<double>(a);
  }
  __device__ __forceinline__ double make_double(const unsigned long int a) {
    return static_cast<double>(a);
  }
  __device__ __forceinline__ double make_double(const long long int a) {
    return static_cast<double>(a);
  }
  __device__ __forceinline__ double make_double(const unsigned long long int a) {
    return static_cast<double>(a);
  }
  __device__ __forceinline__ double make_double(const signed char a) {
    return static_cast<double>(a);
  }
  __device__ __forceinline__ double make_double(const unsigned char a) {
    return static_cast<double>(a);
  }
  __device__ __forceinline__ double make_double(const float a) {
    return static_cast<double>(a);
  }
  __device__ __forceinline__ double make_double(const double a) {
    return static_cast<double>(a);
  }
  __device__ __forceinline__ double make_double(const long double a) {
    return static_cast<double>(a);
  }
  __device__ __forceinline__ double make_double(const hipFloatComplex a) {
    return static_cast<double>(a.x);
  }
  __device__ __forceinline__ double make_double(const hipDoubleComplex a) {
    return static_cast<double>(a.x);
  }
 
  // math functions 
  __device__ __forceinline__ hipFloatComplex conj(const hipFloatComplex& c) {
    return hipConjf(c);
  }
  __device__ __forceinline__ hipDoubleComplex conj(const hipDoubleComplex& z) {
    return hipConj(z);
  }
  __device__ __forceinline__ int nint(const float a) {
    return (a>0.f) ? static_cast<int>(a+0.5f) : static_cast<int>(a-0.5f);
  }
  __device__ __forceinline__ int nint(const double a) {
    return (a>0.) ? static_cast<int>(a+0.5) : static_cast<int>(a-0.5);
  }
  __device__ __forceinline__ float dim(const float a, const float b) {
    const float diff = a-b;
    return (diff>0.f) ? diff : 0.f;
  }
  __device__ __forceinline__ double dim(const double a, const double b) {
    const double diff = a-b;
    return (diff>0.) ? diff : 0.;
  }
} 

#define sign(a,b) copysign(a,b)

#define min2(a,b) min(a,b)
#define min3(aa,ab,ba) min(min(aa,ab),ba)
#define min4(aa,ab,ba,bb) min(min(aa,ab),min(ba,bb))
#define min5(aaa,aab,aba,ba,bb) min(min(min(aaa,aab),aba),min(ba,bb))
#define min6(aaa,aab,aba,baa,bab,bba) min(min(min(aaa,aab),aba),min(min(baa,bab),bba))
#define min7(aaa,aab,aba,abb,baa,bab,bba) min(min(min(aaa,aab),min(aba,abb)),min(min(baa,bab),bba))
#define min8(aaa,aab,aba,abb,baa,bab,bba,bbb) min(min(min(aaa,aab),min(aba,abb)),min(min(baa,bab),min(bba,bbb)))
#define min9(aaaa,aaab,aaba,aba,abb,baa,bab,bba,bbb) min(min(min(min(aaaa,aaab),aaba),min(aba,abb)),min(min(baa,bab),min(bba,bbb)))
#define min10(aaaa,aaab,aaba,aba,abb,baaa,baab,baba,bba,bbb) min(min(min(min(aaaa,aaab),aaba),min(aba,abb)),min(min(min(baaa,baab),baba),min(bba,bbb)))
#define min11(aaaa,aaab,aaba,abaa,abab,abba,baaa,baab,baba,bba,bbb) min(min(min(min(aaaa,aaab),aaba),min(min(abaa,abab),abba)),min(min(min(baaa,baab),baba),min(bba,bbb)))
#define min12(aaaa,aaab,aaba,abaa,abab,abba,baaa,baab,baba,bbaa,bbab,bbba) min(min(min(min(aaaa,aaab),aaba),min(min(abaa,abab),abba)),min(min(min(baaa,baab),baba),min(min(bbaa,bbab),bbba)))
#define min13(aaaa,aaab,aaba,aabb,abaa,abab,abba,baaa,baab,baba,bbaa,bbab,bbba) min(min(min(min(aaaa,aaab),min(aaba,aabb)),min(min(abaa,abab),abba)),min(min(min(baaa,baab),baba),min(min(bbaa,bbab),bbba)))
#define min14(aaaa,aaab,aaba,aabb,abaa,abab,abba,baaa,baab,baba,babb,bbaa,bbab,bbba) min(min(min(min(aaaa,aaab),min(aaba,aabb)),min(min(abaa,abab),abba)),min(min(min(baaa,baab),min(baba,babb)),min(min(bbaa,bbab),bbba)))
#define min15(aaaa,aaab,aaba,aabb,abaa,abab,abba,abbb,baaa,baab,baba,babb,bbaa,bbab,bbba) min(min(min(min(aaaa,aaab),min(aaba,aabb)),min(min(abaa,abab),min(abba,abbb))),min(min(min(baaa,baab),min(baba,babb)),min(min(bbaa,bbab),bbba)))
#define max2(a,b) max(a,b)
#define max3(aa,ab,ba) max(max(aa,ab),ba)
#define max4(aa,ab,ba,bb) max(max(aa,ab),max(ba,bb))
#define max5(aaa,aab,aba,ba,bb) max(max(max(aaa,aab),aba),max(ba,bb))
#define max6(aaa,aab,aba,baa,bab,bba) max(max(max(aaa,aab),aba),max(max(baa,bab),bba))
#define max7(aaa,aab,aba,abb,baa,bab,bba) max(max(max(aaa,aab),max(aba,abb)),max(max(baa,bab),bba))
#define max8(aaa,aab,aba,abb,baa,bab,bba,bbb) max(max(max(aaa,aab),max(aba,abb)),max(max(baa,bab),max(bba,bbb)))
#define max9(aaaa,aaab,aaba,aba,abb,baa,bab,bba,bbb) max(max(max(max(aaaa,aaab),aaba),max(aba,abb)),max(max(baa,bab),max(bba,bbb)))
#define max10(aaaa,aaab,aaba,aba,abb,baaa,baab,baba,bba,bbb) max(max(max(max(aaaa,aaab),aaba),max(aba,abb)),max(max(max(baaa,baab),baba),max(bba,bbb)))
#define max11(aaaa,aaab,aaba,abaa,abab,abba,baaa,baab,baba,bba,bbb) max(max(max(max(aaaa,aaab),aaba),max(max(abaa,abab),abba)),max(max(max(baaa,baab),baba),max(bba,bbb)))
#define max12(aaaa,aaab,aaba,abaa,abab,abba,baaa,baab,baba,bbaa,bbab,bbba) max(max(max(max(aaaa,aaab),aaba),max(max(abaa,abab),abba)),max(max(max(baaa,baab),baba),max(max(bbaa,bbab),bbba)))
#define max13(aaaa,aaab,aaba,aabb,abaa,abab,abba,baaa,baab,baba,bbaa,bbab,bbba) max(max(max(max(aaaa,aaab),max(aaba,aabb)),max(max(abaa,abab),abba)),max(max(max(baaa,baab),baba),max(max(bbaa,bbab),bbba)))
#define max14(aaaa,aaab,aaba,aabb,abaa,abab,abba,baaa,baab,baba,babb,bbaa,bbab,bbba) max(max(max(max(aaaa,aaab),max(aaba,aabb)),max(max(abaa,abab),abba)),max(max(max(baaa,baab),max(baba,babb)),max(max(bbaa,bbab),bbba)))
#define max15(aaaa,aaab,aaba,aabb,abaa,abab,abba,abbb,baaa,baab,baba,babb,bbaa,bbab,bbba) max(max(max(max(aaaa,aaab),max(aaba,aabb)),max(max(abaa,abab),max(abba,abbb))),max(max(max(baaa,baab),max(baba,babb)),max(max(bbaa,bbab),bbba)))
#endif // _GPUFORT_H_