// This file was generated from a template via gpufort --gpufort-create-headers
#ifndef _GPUFORT_ARRAYS_H_
#  define _GPUFORT_ARRAYS_H_
#  include <hip/hip_runtime_api.h>
#  ifndef _GPUFORT_H_
#    include <iostream>
#    include <iomanip>
#    include <stdlib.h>
#    include <vector>
#    define HIP_CHECK(condition)     \
  {                                  \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
  }
#  endif
#ifndef __HIP_DEVICE_COMPILE__
#  include <assert.h>
#endif
#ifndef GPUFORT_ARRAY_PRINT_PREC
#  define GPUFORT_ARRAY_PRINT_PREC 6
#endif

namespace gpufort {
  enum class SyncMode {
    None            = 0, //> No copies between host and device data after initialization and before destruction.
    Copy            = 1, //> Copy to device after initialization and back to host before destruction.
    Copyin          = 2, //> Copy to device after initialization.
    Copyout         = 3, //> Copy from device to host before destruction.
    InvertedCopy    = 4, //> 'Copy'   in opposite direction: Copy from device to host after initialization and back to device before destruction.
    InvertedCopyin  = 5, //> 'Copyin' in opposite direction: Copy from device to host after initialization.
    InvertedCopyout = 6  //> 'Copyout'in opposite direction: Copy from host to device before destruction.
  };

  enum class AllocMode {
    WrapHostWrapDevice         = 0, //> Wrap host and device pointers.
    WrapHostAllocDevice        = 1, //> Wrap host pointer and allocate a new device array.
    AllocHostWrapDevice        = 2, //> Allocate new host array and wrap device pointer.
    AllocHostAllocDevice       = 3, //> Allocate new host and device arrays.
    AllocPinnedHostWrapDevice  = 4, //> Allocate new pinned host array and wrap device pointer.
    AllocPinnedHostAllocDevice = 5  //> Allocate new pinned host array and wrap device pointer.
  };
  
  enum class PrintMode {
    PrintNorms                 = 0, //> Print norms and metrics.
    PrintValues                = 1, //> Print array values; write index in Fortran (i,j,k,...) style.
    PrintValuesAndNorms        = 2, //> Print array values and norms and metrics; write index in Fortran (i,j,k,...) style.
    PrintCValues               = 3, //> Print array values; write index in C/C++ [i] style.
    PrintCValuesAndNorms       = 4  //> Print array values and norms and metrics; write index in C/C++ [i] style.
  };

  /**
   * Intended to be passed as kernel launch parameter.
   * \note Operator `T& operator()` requires parametrization with type T.
   * \note Size of this struct independent of template type T.
   */
  template<typename T>
  struct array_descr1 {
    T*     data_host = nullptr;
    T*     data_dev  = nullptr;
    int num_elements = 0;     //> Number of elements represented by this array.
    int index_offset = -1;    //> Offset for index calculation; scalar product of negative lower bounds and strides.
 
    /**
     * Wrap a host-device pointer pair and store the associated
     * array meta data.
     *
     * \param[in] data_host host data pointer.
     * \param[in] data_dev device data pointer.
     * \param[in] n1,n2,... element counts per dimension of the multi-dim. array 
     * \param[in] lb1,lb2,... lower bounds, i.e. the index of the first element per dimension of the multi-dim. array 
     */ 
    __host__ __device__ void wrap(
        T* data_host,
        T* data_dev,
        const int n1,
        const int lb1
      ) {
       this->data_host = data_host;
       this->data_dev  = data_dev;
       // column-major access
       this->num_elements = 1*n1;
       this->index_offset =
         -lb1*1;
    }
    
    /**
     * Linearize multi-dimensional index.
     *
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ int linearized_index (
      const int i1
    ) const {
      return this->index_offset
          + i1*1;
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ T& operator() (
      const int i1
    ) {
      const int index = linearized_index(
        i1
      );
      #ifdef __HIP_DEVICE_COMPILE__
      return this->data_dev[index];
      #else
      assert(i1 >= lbound(1));
      assert(i1 <= ubound(1));
      return this->data_host[index];
      #endif
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ const T& operator() (
      const int i1
    ) const {
      const int index = linearized_index(
        i1
      );
      #ifdef __HIP_DEVICE_COMPILE__
      return this->data_dev[index];
      #else
      assert(i1 >= lbound(1));
      assert(i1 <= ubound(1));
      return this->data_host[index];
      #endif
    }
    
    /**
     * \return Size of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,1
     */
    __host__ __device__ __forceinline__ int size(int dim) const {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(dim >= 1);
      assert(dim <= 1);
      #endif
      switch(dim) {
       case 1:
	        return this->num_elements / 1;
       default:
          #ifndef __HIP_DEVICE_COMPILE__
          std::cerr << "‘dim’ argument of ‘gpufort::array1::size’ is not a valid dimension index ('dim': "<<dim<<", max dimension: 1" << std::endl;
          std::terminate();
          #else
          return -1;
          #endif
      }
    }
    
    /**
     * \return Lower bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,1
     */
    __host__ __device__ __forceinline__ int lbound(int dim) const {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(dim >= 1);
      assert(dim <= 1);
      #endif
      int offset_remainder = this->index_offset;
      int lb = -offset_remainder / 1;
      #ifndef __HIP_DEVICE_COMPILE__
      offset_remainder = offset_remainder + lb*1;
      assert(offset_remainder == 0);
      #endif
      return lb;
    }
    
    /**
     * \return Upper bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,1
     */
    __host__ __device__ __forceinline__ int ubound(int dim) const {
      return this->lbound(dim) + this->size(dim) - 1;
    }
  };

  template<typename T>
  struct array1{
    array_descr1<T> data;
    AllocMode              alloc_mode = AllocMode::WrapHostAllocDevice; //> Data allocation strategy. Default: 
                                                                        //> wrap the host and allocate device data
    SyncMode               sync_mode  = SyncMode::None;                 //> How data should be synchronized
                                                                        //> during the initialization and destruction of this GPUFORT array.
    int num_refs             = 0;  //> Number of references.
    size_t bytes_per_element = -1; //> Bytes per element; stored to make num_data_bytes routine independent of T 

    array1() {
      // do nothing
    }
    
    ~array1() {
      // do nothing
    }

    /**
     * Initialize.
     * \param[in] data_host host data pointer (may be nullptr; see the note).
     * \param[in] pinned the host array is pinned or should be pinned.
     * \param[in] copyout_at_destruction copy data out from the
     *            device to the host when this instance gets destroyed.
     * \note Allocates a new host_data pointer if the
     * passed pointer is a nullptr. If the pinned argument is true,
     * the new hostpotr is allocated via hipHostMalloc.
     * Otherwise, it is allocated via classic malloc.
     */
    __host__ hipError_t init_async(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int n1,
        int lb1,
        hipStream_t stream,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      this->data.wrap(
          data_host,
          data_dev,
          n1,
          lb1
      );
      this->bytes_per_element = bytes_per_element;
      this->num_refs          = 1;
      this->alloc_mode        = alloc_mode;
      this->sync_mode         = sync_mode;
  
      // host array allocation
      hipError_t ierr = hipSuccess;
      switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
#endif
        case AllocMode::AllocPinnedHostWrapDevice:
        case AllocMode::AllocPinnedHostAllocDevice:
          ierr = hipHostMalloc((void**) &this->data.data_host,this->num_data_bytes(),0);
	  break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
          this->data.data_host = (T*) malloc(this->num_data_bytes());
	  break;
#endif
	case AllocMode::WrapHostWrapDevice:
        case AllocMode::WrapHostAllocDevice:
          // do nothing as already set by data.wrap call
	  break;
	default:
          std::cerr << "ERROR: gpufort::array1::init_async(...): Unexpected value for 'this->alloc_mode': " 
                    << static_cast<int>(this->alloc_mode) << std::endl; 
          std::terminate();
          break;
      }
      // device array allocation
      if ( ierr == hipSuccess ) {
        switch (this->alloc_mode) {
          case AllocMode::AllocHostWrapDevice:
          case AllocMode::AllocPinnedHostWrapDevice:
          case AllocMode::WrapHostWrapDevice:
            // do nothing as already set by data.wrap call
            break;
          case AllocMode::WrapHostAllocDevice:
          case AllocMode::AllocHostAllocDevice:
          case AllocMode::AllocPinnedHostAllocDevice:
            ierr = hipMalloc((void**) &this->data.data_dev, this->num_data_bytes());
            break;
          default:
            break;
        }
      }
      if ( ierr == hipSuccess ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::None:
          case SyncMode::Copyout:
          case SyncMode::InvertedCopyout:
            // do nothing
            break;
          case SyncMode::Copy:
          case SyncMode::Copyin:
            ierr = this->copy_to_device_async(stream);
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyin:
            ierr = this->copy_to_host_async(stream);
            break;
          default:
            std::cerr << "ERROR: gpufort::array1::init_async(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
      }
      return ierr;
    }
   
    __host__ hipError_t init_async(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int* sizes,
        int* lower_bounds,
        hipStream_t stream,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      return this->init_async(
        bytes_per_element,
        data_host,
        data_dev,
        sizes[0],
        lower_bounds[0],
        stream,alloc_mode,sync_mode);
    }
    
    /**
     * Destroy associated host and device data
     * if this is not a wrapper. Depending
     * on the pinned attribue, hipHostFree 
     * is used instead of free to deallocate
     * the host data.
     */
    __host__ hipError_t destroy_async(hipStream_t stream) {
      hipError_t ierr = hipSuccess;
      if ( bytes_per_element > 0 ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::Copy:
          case SyncMode::Copyout:
            ierr = this->copy_to_host_async(stream);
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyout:
            ierr = this->copy_to_device_async(stream);
            break;
          case SyncMode::None:
          case SyncMode::Copyin:
          case SyncMode::InvertedCopyin:
            // do nothing
            break;
          default:
            std::cerr << "ERROR: gpufort::array1::destroy_async(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
        // host array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
#endif
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipHostFree(this->data.data_host);
              break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
              free(this->data.data_host); 
              break;
#endif
            case AllocMode::WrapHostWrapDevice:
            case AllocMode::WrapHostAllocDevice:
              break;
            default:
              std::cerr << "ERROR: gpufort::array1::destroy_async(...): Unexpected value for 'alloc_mode': " 
                        << static_cast<int>(alloc_mode) << std::endl; 
              std::terminate();
              break;
          }
        }
        // device array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::WrapHostWrapDevice:
              // do nothing as already set by data.wrap call
              break;
            case AllocMode::WrapHostAllocDevice:
            case AllocMode::AllocHostAllocDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipFree(this->data.data_dev);
              break;
            default:
              break;
          }
        }
        if ( ierr == hipSuccess ) {
          this->data.data_host    = nullptr;
          this->data.data_dev     = nullptr;
          this->bytes_per_element = -1; 
        } 
      }
      return ierr;
    }
    
    /**
     * Copy data to a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the destinaton buffer is interpreted as host or device variable
     * and the source is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_to_buffer_async(
       void* buffer, hipMemcpyKind memcpy_kind,
       hipStream_t stream
    ) {
      T* src = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyDeviceToHost: 
        case hipMemcpyDeviceToDevice:
          src = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyHostToDevice:
          src = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array1::copy_to_buffer_async(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(src!=nullptr);
      #endif
      return hipMemcpyAsync(
        buffer, 
        (void*) src,
        this->num_data_bytes(), 
        memcpy_kind, stream);
    }
    
    /**
     * Copy data from a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the source buffer is interpreted as host or device variable
     * and the destination is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_from_buffer_async(
       void* buffer, hipMemcpyKind memcpy_kind,
       hipStream_t stream
    ) {
      T* dest = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyHostToDevice: 
        case hipMemcpyDeviceToDevice:
          dest = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyDeviceToHost:
          dest = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array1::copy_from_buffer_async(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(dest!=nullptr);
      #endif
      return hipMemcpyAsync(
        (void*) dest,
        buffer, 
        this->num_data_bytes(), 
        memcpy_kind, stream);
    }
    
    /**
     * Copy host data to the device.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_host_async(hipStream_t stream) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpyAsync(
        (void*) this->data.data_host, 
        (void*) this->data.data_dev,
        this->num_data_bytes(), 
        hipMemcpyDeviceToHost, stream);
    }
    
    /**
     * Copy device data to the host.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_device_async(hipStream_t stream) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpyAsync(
        (void*) this->data.data_dev, 
        (void*) this->data.data_host,
        this->num_data_bytes(),
        hipMemcpyHostToDevice, stream);
    }
    
    /** 
     * Copies the struct to the device into an pre-allocated
     * device memory block but does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[in] device_struct device memory address to copy to
     */
    __host__ hipError_t copy_self_to_device_async(
        gpufort::array1<T>* device_struct,
        hipStream_t stream
    ) {
      const size_t size = sizeof(array1<char>); // sizeof(T*) = sizeof(char*)
      return hipMemcpyAsync(
          (void*) device_struct, 
          (void*) this,
          size,
          hipMemcpyHostToDevice, stream);
    }

    /** 
     * Copies the struct to the device but
     * does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[inout] device_copy pointer to device copy pointer 
     */
    __host__ hipError_t create_device_copy_async(
        gpufort::array1<T>** device_copy,
        hipStream_t stream
    ) {
      const size_t size = sizeof(array1<char>); // sizeof(T*) = sizeof(char*)
      hipError_t ierr   = hipMalloc((void**)device_copy,size);      
      if ( ierr == hipSuccess ) {
        return this->copy_self_to_device_async(*device_copy,stream);
      } else {
        return ierr;
      }
    }
    /**
     * Initialize.
     * \param[in] data_host host data pointer (may be nullptr; see the note).
     * \param[in] pinned the host array is pinned or should be pinned.
     * \param[in] copyout_at_destruction copy data out from the
     *            device to the host when this instance gets destroyed.
     * \note Allocates a new host_data pointer if the
     * passed pointer is a nullptr. If the pinned argument is true,
     * the new hostpotr is allocated via hipHostMalloc.
     * Otherwise, it is allocated via classic malloc.
     */
    __host__ hipError_t init(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int n1,
        int lb1,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      this->data.wrap(
          data_host,
          data_dev,
          n1,
          lb1
      );
      this->bytes_per_element = bytes_per_element;
      this->num_refs          = 1;
      this->alloc_mode        = alloc_mode;
      this->sync_mode         = sync_mode;
  
      // host array allocation
      hipError_t ierr = hipSuccess;
      switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
#endif
        case AllocMode::AllocPinnedHostWrapDevice:
        case AllocMode::AllocPinnedHostAllocDevice:
          ierr = hipHostMalloc((void**) &this->data.data_host,this->num_data_bytes(),0);
	  break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
          this->data.data_host = (T*) malloc(this->num_data_bytes());
	  break;
#endif
	case AllocMode::WrapHostWrapDevice:
        case AllocMode::WrapHostAllocDevice:
          // do nothing as already set by data.wrap call
	  break;
	default:
          std::cerr << "ERROR: gpufort::array1::init(...): Unexpected value for 'this->alloc_mode': " 
                    << static_cast<int>(this->alloc_mode) << std::endl; 
          std::terminate();
          break;
      }
      // device array allocation
      if ( ierr == hipSuccess ) {
        switch (this->alloc_mode) {
          case AllocMode::AllocHostWrapDevice:
          case AllocMode::AllocPinnedHostWrapDevice:
          case AllocMode::WrapHostWrapDevice:
            // do nothing as already set by data.wrap call
            break;
          case AllocMode::WrapHostAllocDevice:
          case AllocMode::AllocHostAllocDevice:
          case AllocMode::AllocPinnedHostAllocDevice:
            ierr = hipMalloc((void**) &this->data.data_dev, this->num_data_bytes());
            break;
          default:
            break;
        }
      }
      if ( ierr == hipSuccess ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::None:
          case SyncMode::Copyout:
          case SyncMode::InvertedCopyout:
            // do nothing
            break;
          case SyncMode::Copy:
          case SyncMode::Copyin:
            ierr = this->copy_to_device();
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyin:
            ierr = this->copy_to_host();
            break;
          default:
            std::cerr << "ERROR: gpufort::array1::init(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
      }
      return ierr;
    }
   
    __host__ hipError_t init(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int* sizes,
        int* lower_bounds,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      return this->init(
        bytes_per_element,
        data_host,
        data_dev,
        sizes[0],
        lower_bounds[0],alloc_mode,sync_mode);
    }
    
    /**
     * Destroy associated host and device data
     * if this is not a wrapper. Depending
     * on the pinned attribue, hipHostFree 
     * is used instead of free to deallocate
     * the host data.
     */
    __host__ hipError_t destroy() {
      hipError_t ierr = hipSuccess;
      if ( bytes_per_element > 0 ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::Copy:
          case SyncMode::Copyout:
            ierr = this->copy_to_host();
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyout:
            ierr = this->copy_to_device();
            break;
          case SyncMode::None:
          case SyncMode::Copyin:
          case SyncMode::InvertedCopyin:
            // do nothing
            break;
          default:
            std::cerr << "ERROR: gpufort::array1::destroy(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
        // host array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
#endif
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipHostFree(this->data.data_host);
              break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
              free(this->data.data_host); 
              break;
#endif
            case AllocMode::WrapHostWrapDevice:
            case AllocMode::WrapHostAllocDevice:
              break;
            default:
              std::cerr << "ERROR: gpufort::array1::destroy(...): Unexpected value for 'alloc_mode': " 
                        << static_cast<int>(alloc_mode) << std::endl; 
              std::terminate();
              break;
          }
        }
        // device array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::WrapHostWrapDevice:
              // do nothing as already set by data.wrap call
              break;
            case AllocMode::WrapHostAllocDevice:
            case AllocMode::AllocHostAllocDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipFree(this->data.data_dev);
              break;
            default:
              break;
          }
        }
        if ( ierr == hipSuccess ) {
          this->data.data_host    = nullptr;
          this->data.data_dev     = nullptr;
          this->bytes_per_element = -1; 
        } 
      }
      return ierr;
    }
    
    /**
     * Copy data to a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the destinaton buffer is interpreted as host or device variable
     * and the source is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_to_buffer(
       void* buffer, hipMemcpyKind memcpy_kind
    ) {
      T* src = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyDeviceToHost: 
        case hipMemcpyDeviceToDevice:
          src = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyHostToDevice:
          src = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array1::copy_to_buffer(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(src!=nullptr);
      #endif
      return hipMemcpy(
        buffer, 
        (void*) src,
        this->num_data_bytes(), 
        memcpy_kind);
    }
    
    /**
     * Copy data from a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the source buffer is interpreted as host or device variable
     * and the destination is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_from_buffer(
       void* buffer, hipMemcpyKind memcpy_kind
    ) {
      T* dest = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyHostToDevice: 
        case hipMemcpyDeviceToDevice:
          dest = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyDeviceToHost:
          dest = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array1::copy_from_buffer(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(dest!=nullptr);
      #endif
      return hipMemcpy(
        (void*) dest,
        buffer, 
        this->num_data_bytes(), 
        memcpy_kind);
    }
    
    /**
     * Copy host data to the device.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_host() {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpy(
        (void*) this->data.data_host, 
        (void*) this->data.data_dev,
        this->num_data_bytes(), 
        hipMemcpyDeviceToHost);
    }
    
    /**
     * Copy device data to the host.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_device() {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpy(
        (void*) this->data.data_dev, 
        (void*) this->data.data_host,
        this->num_data_bytes(),
        hipMemcpyHostToDevice);
    }
    
    /** 
     * Copies the struct to the device into an pre-allocated
     * device memory block but does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[in] device_struct device memory address to copy to
     */
    __host__ hipError_t copy_self_to_device(
        gpufort::array1<T>* device_struct
    ) {
      const size_t size = sizeof(array1<char>); // sizeof(T*) = sizeof(char*)
      return hipMemcpy(
          (void*) device_struct, 
          (void*) this,
          size,
          hipMemcpyHostToDevice);
    }

    /** 
     * Copies the struct to the device but
     * does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[inout] device_copy pointer to device copy pointer 
     */
    __host__ hipError_t create_device_copy(
        gpufort::array1<T>** device_copy
    ) {
      const size_t size = sizeof(array1<char>); // sizeof(T*) = sizeof(char*)
      hipError_t ierr   = hipMalloc((void**)device_copy,size);      
      if ( ierr == hipSuccess ) {
        return this->copy_self_to_device(*device_copy);
      } else {
        return ierr;
      }
    }

    /**
     * Allocate a host buffer with
     * the same size as the data buffers
     * associated with this gpufort array.
     * @see num_data_bytes()
     * \param[inout] pointer to the buffer to allocate
     * \param[in] pinned If the memory should be pinned (default=True)
     * \param[in] flags  Flags for the host memory allocation (default=0).
     */
    __host__ hipError_t allocate_host_buffer(void** buffer,bool pinned=true,int flags=0) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
      if ( pinned ) { 
        ierr = hipHostMalloc(buffer,this->num_data_bytes(),flags);
      } else {
        *buffer = malloc(this->num_data_bytes());
      }
      return ierr;
    }

    /**
     * Deallocate a host buffer
     * created via the allocate_host_buffer routine.
     * @see num_data_bytes(), allocate_host_buffer
     * \param[inout] the buffer to deallocte
     * \param[in] pinned If the memory to deallocate is pinned (default=True)
     */
    __host__ hipError_t deallocate_host_buffer(void* buffer,bool pinned=true) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
      if ( pinned ) { 
        ierr = hipHostFree(buffer);
      } else {
        free(buffer);
      }
      return ierr;
    }
    /**
     * Allocate a device buffer with
     * the same size as the data buffers
     * associated with this gpufort array.
     * @see num_data_bytes()
     * \param[inout] pointer to the buffer to allocate
     */
    __host__ hipError_t allocate_device_buffer(void** buffer) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
 
      ierr = hipMalloc(buffer,this->num_data_bytes());
      return ierr;
    }

    /**
     * Deallocate a device buffer
     * created via the allocate_device_buffer routine.
     * @see num_data_bytes(), allocate_device_buffer
     * \param[inout] the buffer to deallocte
     */
    __host__ hipError_t deallocate_device_buffer(void* buffer) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
 
      ierr = hipFree(buffer);
      return ierr;
    }

    /** 
     * Copy metadata from another gpufort array to this gpufort array.
     * \note The allocation mode is always set to AllocMode::WrapHostWrapDevice
     *       and the sync mode is set to SyncMode::None.
     * \note No deep copy, i.e. the host and device data is not copied, just the pointers
     * to the data.
     */
    __host__ void copy(const gpufort::array1<T>& other) {
      this->init(
        other.bytes_per_element,
        other.data.data_host,
        other.data.data_dev,
        other.size(1),
        other.lbound(1),
        AllocMode::WrapHostWrapDevice,
        SyncMode::None);
    }

    /** 
     *  \return Number of bytes required to store 
     *  the array elements on the host or the device.
     */ 
    __host__ size_t num_data_bytes() {
      return this->data.num_elements * this->bytes_per_element;
    }
    
    /**
     * \return the number of elements of this array.
     */
    __host__ int num_elements() {
      return this->data.num_elements;
    }

    /**
     * Linearize multi-dimensional index.
     *
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ int linearized_index (
      const int i1
    ) const {
      return this->data.linearized_index(
        i1
      );
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ T& operator() (
      const int i1
    ) {
      return this->data(
        i1
      );
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ const T& operator() (
      const int i1
    ) const {
      return this->data(
        i1
      );
    }
    
    /**
     * \return Size of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,1
     */
    __host__ __device__ __forceinline__ int size(int dim) const {
      return this->data.size(dim);
    }
    
    /**
     * \return Lower bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,1
     */
    __host__ __device__ __forceinline__ int lbound(int dim) const {
      return this->data.lbound(dim);
    }
    
    /**
     * \return Upper bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,1
     */
    __host__ __device__ __forceinline__ int ubound(int dim) const {
      return this->data.ubound(dim);
    }
    


    /**
     * Write host data values and their corresponding index
     * to an output stream.
     * \param[inout] out        output stream
     * \param[in]    prefix     prefix to put before each output line
     * \param[in]    print_prec precision to use when printing floating point numbers 
     *               [default=GPUFORT_ARRAY_PRINT_PREC[default=6]]
     * \note This method only makes sense for numeric data types and
     *       may result in compilation, runtime errors or undefined output if the
     *       underlying data is not numeric.
     */
    __host__ void print_host_data(
        std::ostream& out, const char* prefix, PrintMode print_mode,
        const int print_prec=GPUFORT_ARRAY_PRINT_PREC) const {
      const bool print_norms = 
	print_mode==PrintMode::PrintNorms ||
        print_mode==PrintMode::PrintValuesAndNorms ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      const bool print_values = 
        print_mode==PrintMode::PrintValues ||
        print_mode==PrintMode::PrintValuesAndNorms;
      const bool print_values_c_style = 
        print_mode==PrintMode::PrintCValues ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      if ( !print_norms && !print_values && !print_values_c_style ) {
        std::cerr << "ERROR: gpufort::array1::print_host_data(...): Unexpected value for 'print_mode': " 
                  << static_cast<int>(print_mode) << std::endl; 
        std::terminate();
      }
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1); 
      const int n = this->data.num_elements; 
      T* A_h   = this->data.data_host;
      T min = +std::numeric_limits<T>::max();
      T max = -std::numeric_limits<T>::max();
      T sum = 0;
      T l1  = 0;
      T l2  = 0;
      out << prefix << ":\n";
    for ( int i1 = 0; i1 < n1; i1++ ) {
      const int idx = this->linearized_index(i1);
      T value = A_h[idx];
      if ( print_norms ) {
        min  = std::min(value,min);
        max  = std::max(value,max);
        sum += value;
        l1  += std::abs(value);
        l2  += value*value;
      }
      if ( print_values ) {
          out << prefix << "(" << (lb1+i1) <<  ") = " << std::setprecision(print_prec) << value << "\n";
      } else if ( print_values_c_style ) {
          out << prefix << "[" << idx << "] = " << std::setprecision(print_prec) << value << "\n";
      }
    } // for loops
    if ( print_norms ) {
      out << prefix << ":min=" << std::setprecision(print_prec) << min << "\n";
      out << prefix << ":max=" << std::setprecision(print_prec) << max << "\n";
      out << prefix << ":sum=" << std::setprecision(print_prec) << sum << "\n";
      out << prefix << ":l1="  << std::setprecision(print_prec) << l1  << "\n";
      out << prefix << ":l2="  << std::setprecision(print_prec) << std::sqrt(l2) << "\n";
      out << prefix << "num_elements=" << std::setprecision(print_prec) << n << "\n";
    }
  }
    /**
     * Write device data values and their corresponding index
     * to an output stream.
     * \param[inout] out        output stream
     * \param[in]    prefix     prefix to put before each output line
     * \param[in]    print_prec precision to use when printing floating point numbers 
     *               [default=GPUFORT_ARRAY_PRINT_PREC[default=6]]
     * \note This method only makes sense for numeric data types and
     *       may result in compilation, runtime errors or undefined output if the
     *       underlying data is not numeric.
     */
    __host__ void print_device_data(
        std::ostream& out, const char* prefix, PrintMode print_mode,
        const int print_prec=GPUFORT_ARRAY_PRINT_PREC) const {
      const bool print_norms = 
	print_mode==PrintMode::PrintNorms ||
        print_mode==PrintMode::PrintValuesAndNorms ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      const bool print_values = 
        print_mode==PrintMode::PrintValues ||
        print_mode==PrintMode::PrintValuesAndNorms;
      const bool print_values_c_style = 
        print_mode==PrintMode::PrintCValues ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      if ( !print_norms && !print_values && !print_values_c_style ) {
        std::cerr << "ERROR: gpufort::array1::print_device_data(...): Unexpected value for 'print_mode': " 
                  << static_cast<int>(print_mode) << std::endl; 
        std::terminate();
      }
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1); 
      const int n = this->data.num_elements; 
      std::vector<T> host_array(n);
      T* A_h = host_array.data();
      T* A   = this->data.data_dev;
      HIP_CHECK(hipMemcpy(A_h, A, n*sizeof(T), hipMemcpyDeviceToHost));
      T min = +std::numeric_limits<T>::max();
      T max = -std::numeric_limits<T>::max();
      T sum = 0;
      T l1  = 0;
      T l2  = 0;
      out << prefix << ":\n";
    for ( int i1 = 0; i1 < n1; i1++ ) {
      const int idx = this->linearized_index(i1);
      T value = A_h[idx];
      if ( print_norms ) {
        min  = std::min(value,min);
        max  = std::max(value,max);
        sum += value;
        l1  += std::abs(value);
        l2  += value*value;
      }
      if ( print_values ) {
          out << prefix << "(" << (lb1+i1) <<  ") = " << std::setprecision(print_prec) << value << "\n";
      } else if ( print_values_c_style ) {
          out << prefix << "[" << idx << "] = " << std::setprecision(print_prec) << value << "\n";
      }
    } // for loops
    if ( print_norms ) {
      out << prefix << ":min=" << std::setprecision(print_prec) << min << "\n";
      out << prefix << ":max=" << std::setprecision(print_prec) << max << "\n";
      out << prefix << ":sum=" << std::setprecision(print_prec) << sum << "\n";
      out << prefix << ":l1="  << std::setprecision(print_prec) << l1  << "\n";
      out << prefix << ":l2="  << std::setprecision(print_prec) << std::sqrt(l2) << "\n";
      out << prefix << "num_elements=" << std::setprecision(print_prec) << n << "\n";
    }
  }

  };

  /**
   * Intended to be passed as kernel launch parameter.
   * \note Operator `T& operator()` requires parametrization with type T.
   * \note Size of this struct independent of template type T.
   */
  template<typename T>
  struct array_descr2 {
    T*     data_host = nullptr;
    T*     data_dev  = nullptr;
    int num_elements = 0;     //> Number of elements represented by this array.
    int index_offset = -1;    //> Offset for index calculation; scalar product of negative lower bounds and strides.
    int    stride2  = -1;    //> Stride 2 for linearizing 2-dimensional index.
 
    /**
     * Wrap a host-device pointer pair and store the associated
     * array meta data.
     *
     * \param[in] data_host host data pointer.
     * \param[in] data_dev device data pointer.
     * \param[in] n1,n2,... element counts per dimension of the multi-dim. array 
     * \param[in] lb1,lb2,... lower bounds, i.e. the index of the first element per dimension of the multi-dim. array 
     */ 
    __host__ __device__ void wrap(
        T* data_host,
        T* data_dev,
        const int n1,
        const int n2,
        const int lb1,
        const int lb2
      ) {
       this->data_host = data_host;
       this->data_dev  = data_dev;
       // column-major access
       this->stride2 = 1*n1;
       this->num_elements = this->stride2*n2;
       this->index_offset =
         -lb1*1
         -lb2*this->stride2;
    }
    
    /**
     * Linearize multi-dimensional index.
     *
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ int linearized_index (
      const int i1,
      const int i2
    ) const {
      return this->index_offset
          + i1*1
          + i2*this->stride2;
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ T& operator() (
      const int i1,
      const int i2
    ) {
      const int index = linearized_index(
        i1,i2
      );
      #ifdef __HIP_DEVICE_COMPILE__
      return this->data_dev[index];
      #else
      assert(i1 >= lbound(1));
      assert(i1 <= ubound(1));
      assert(i2 >= lbound(2));
      assert(i2 <= ubound(2));
      return this->data_host[index];
      #endif
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ const T& operator() (
      const int i1,
      const int i2
    ) const {
      const int index = linearized_index(
        i1,i2
      );
      #ifdef __HIP_DEVICE_COMPILE__
      return this->data_dev[index];
      #else
      assert(i1 >= lbound(1));
      assert(i1 <= ubound(1));
      assert(i2 >= lbound(2));
      assert(i2 <= ubound(2));
      return this->data_host[index];
      #endif
    }
    
    /**
     * \return Size of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,2
     */
    __host__ __device__ __forceinline__ int size(int dim) const {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(dim >= 1);
      assert(dim <= 2);
      #endif
      switch(dim) {
       case 2:
	        return this->num_elements / this->stride2;
       case 1:
	        return this->stride2 / 1;
       default:
          #ifndef __HIP_DEVICE_COMPILE__
          std::cerr << "‘dim’ argument of ‘gpufort::array2::size’ is not a valid dimension index ('dim': "<<dim<<", max dimension: 2" << std::endl;
          std::terminate();
          #else
          return -1;
          #endif
      }
    }
    
    /**
     * \return Lower bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,2
     */
    __host__ __device__ __forceinline__ int lbound(int dim) const {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(dim >= 1);
      assert(dim <= 2);
      #endif
      int offset_remainder = this->index_offset;
      int lb = -offset_remainder / this->stride2;
      if ( dim == 2 ) return lb;
      offset_remainder = offset_remainder + lb*this->stride2;
      lb = -offset_remainder / 1;
      #ifndef __HIP_DEVICE_COMPILE__
      offset_remainder = offset_remainder + lb*1;
      assert(offset_remainder == 0);
      #endif
      return lb;
    }
    
    /**
     * \return Upper bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,2
     */
    __host__ __device__ __forceinline__ int ubound(int dim) const {
      return this->lbound(dim) + this->size(dim) - 1;
    }
  };

  template<typename T>
  struct array2{
    array_descr2<T> data;
    AllocMode              alloc_mode = AllocMode::WrapHostAllocDevice; //> Data allocation strategy. Default: 
                                                                        //> wrap the host and allocate device data
    SyncMode               sync_mode  = SyncMode::None;                 //> How data should be synchronized
                                                                        //> during the initialization and destruction of this GPUFORT array.
    int num_refs             = 0;  //> Number of references.
    size_t bytes_per_element = -1; //> Bytes per element; stored to make num_data_bytes routine independent of T 

    array2() {
      // do nothing
    }
    
    ~array2() {
      // do nothing
    }

    /**
     * Initialize.
     * \param[in] data_host host data pointer (may be nullptr; see the note).
     * \param[in] pinned the host array is pinned or should be pinned.
     * \param[in] copyout_at_destruction copy data out from the
     *            device to the host when this instance gets destroyed.
     * \note Allocates a new host_data pointer if the
     * passed pointer is a nullptr. If the pinned argument is true,
     * the new hostpotr is allocated via hipHostMalloc.
     * Otherwise, it is allocated via classic malloc.
     */
    __host__ hipError_t init_async(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int n1,int n2,
        int lb1,int lb2,
        hipStream_t stream,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      this->data.wrap(
          data_host,
          data_dev,
          n1,n2,
          lb1,lb2
      );
      this->bytes_per_element = bytes_per_element;
      this->num_refs          = 1;
      this->alloc_mode        = alloc_mode;
      this->sync_mode         = sync_mode;
  
      // host array allocation
      hipError_t ierr = hipSuccess;
      switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
#endif
        case AllocMode::AllocPinnedHostWrapDevice:
        case AllocMode::AllocPinnedHostAllocDevice:
          ierr = hipHostMalloc((void**) &this->data.data_host,this->num_data_bytes(),0);
	  break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
          this->data.data_host = (T*) malloc(this->num_data_bytes());
	  break;
#endif
	case AllocMode::WrapHostWrapDevice:
        case AllocMode::WrapHostAllocDevice:
          // do nothing as already set by data.wrap call
	  break;
	default:
          std::cerr << "ERROR: gpufort::array2::init_async(...): Unexpected value for 'this->alloc_mode': " 
                    << static_cast<int>(this->alloc_mode) << std::endl; 
          std::terminate();
          break;
      }
      // device array allocation
      if ( ierr == hipSuccess ) {
        switch (this->alloc_mode) {
          case AllocMode::AllocHostWrapDevice:
          case AllocMode::AllocPinnedHostWrapDevice:
          case AllocMode::WrapHostWrapDevice:
            // do nothing as already set by data.wrap call
            break;
          case AllocMode::WrapHostAllocDevice:
          case AllocMode::AllocHostAllocDevice:
          case AllocMode::AllocPinnedHostAllocDevice:
            ierr = hipMalloc((void**) &this->data.data_dev, this->num_data_bytes());
            break;
          default:
            break;
        }
      }
      if ( ierr == hipSuccess ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::None:
          case SyncMode::Copyout:
          case SyncMode::InvertedCopyout:
            // do nothing
            break;
          case SyncMode::Copy:
          case SyncMode::Copyin:
            ierr = this->copy_to_device_async(stream);
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyin:
            ierr = this->copy_to_host_async(stream);
            break;
          default:
            std::cerr << "ERROR: gpufort::array2::init_async(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
      }
      return ierr;
    }
   
    __host__ hipError_t init_async(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int* sizes,
        int* lower_bounds,
        hipStream_t stream,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      return this->init_async(
        bytes_per_element,
        data_host,
        data_dev,
        sizes[0],sizes[1],
        lower_bounds[0],lower_bounds[1],
        stream,alloc_mode,sync_mode);
    }
    
    /**
     * Destroy associated host and device data
     * if this is not a wrapper. Depending
     * on the pinned attribue, hipHostFree 
     * is used instead of free to deallocate
     * the host data.
     */
    __host__ hipError_t destroy_async(hipStream_t stream) {
      hipError_t ierr = hipSuccess;
      if ( bytes_per_element > 0 ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::Copy:
          case SyncMode::Copyout:
            ierr = this->copy_to_host_async(stream);
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyout:
            ierr = this->copy_to_device_async(stream);
            break;
          case SyncMode::None:
          case SyncMode::Copyin:
          case SyncMode::InvertedCopyin:
            // do nothing
            break;
          default:
            std::cerr << "ERROR: gpufort::array2::destroy_async(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
        // host array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
#endif
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipHostFree(this->data.data_host);
              break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
              free(this->data.data_host); 
              break;
#endif
            case AllocMode::WrapHostWrapDevice:
            case AllocMode::WrapHostAllocDevice:
              break;
            default:
              std::cerr << "ERROR: gpufort::array2::destroy_async(...): Unexpected value for 'alloc_mode': " 
                        << static_cast<int>(alloc_mode) << std::endl; 
              std::terminate();
              break;
          }
        }
        // device array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::WrapHostWrapDevice:
              // do nothing as already set by data.wrap call
              break;
            case AllocMode::WrapHostAllocDevice:
            case AllocMode::AllocHostAllocDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipFree(this->data.data_dev);
              break;
            default:
              break;
          }
        }
        if ( ierr == hipSuccess ) {
          this->data.data_host    = nullptr;
          this->data.data_dev     = nullptr;
          this->bytes_per_element = -1; 
        } 
      }
      return ierr;
    }
    
    /**
     * Copy data to a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the destinaton buffer is interpreted as host or device variable
     * and the source is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_to_buffer_async(
       void* buffer, hipMemcpyKind memcpy_kind,
       hipStream_t stream
    ) {
      T* src = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyDeviceToHost: 
        case hipMemcpyDeviceToDevice:
          src = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyHostToDevice:
          src = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array2::copy_to_buffer_async(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(src!=nullptr);
      #endif
      return hipMemcpyAsync(
        buffer, 
        (void*) src,
        this->num_data_bytes(), 
        memcpy_kind, stream);
    }
    
    /**
     * Copy data from a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the source buffer is interpreted as host or device variable
     * and the destination is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_from_buffer_async(
       void* buffer, hipMemcpyKind memcpy_kind,
       hipStream_t stream
    ) {
      T* dest = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyHostToDevice: 
        case hipMemcpyDeviceToDevice:
          dest = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyDeviceToHost:
          dest = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array2::copy_from_buffer_async(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(dest!=nullptr);
      #endif
      return hipMemcpyAsync(
        (void*) dest,
        buffer, 
        this->num_data_bytes(), 
        memcpy_kind, stream);
    }
    
    /**
     * Copy host data to the device.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_host_async(hipStream_t stream) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpyAsync(
        (void*) this->data.data_host, 
        (void*) this->data.data_dev,
        this->num_data_bytes(), 
        hipMemcpyDeviceToHost, stream);
    }
    
    /**
     * Copy device data to the host.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_device_async(hipStream_t stream) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpyAsync(
        (void*) this->data.data_dev, 
        (void*) this->data.data_host,
        this->num_data_bytes(),
        hipMemcpyHostToDevice, stream);
    }
    
    /** 
     * Copies the struct to the device into an pre-allocated
     * device memory block but does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[in] device_struct device memory address to copy to
     */
    __host__ hipError_t copy_self_to_device_async(
        gpufort::array2<T>* device_struct,
        hipStream_t stream
    ) {
      const size_t size = sizeof(array2<char>); // sizeof(T*) = sizeof(char*)
      return hipMemcpyAsync(
          (void*) device_struct, 
          (void*) this,
          size,
          hipMemcpyHostToDevice, stream);
    }

    /** 
     * Copies the struct to the device but
     * does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[inout] device_copy pointer to device copy pointer 
     */
    __host__ hipError_t create_device_copy_async(
        gpufort::array2<T>** device_copy,
        hipStream_t stream
    ) {
      const size_t size = sizeof(array2<char>); // sizeof(T*) = sizeof(char*)
      hipError_t ierr   = hipMalloc((void**)device_copy,size);      
      if ( ierr == hipSuccess ) {
        return this->copy_self_to_device_async(*device_copy,stream);
      } else {
        return ierr;
      }
    }
    /**
     * Initialize.
     * \param[in] data_host host data pointer (may be nullptr; see the note).
     * \param[in] pinned the host array is pinned or should be pinned.
     * \param[in] copyout_at_destruction copy data out from the
     *            device to the host when this instance gets destroyed.
     * \note Allocates a new host_data pointer if the
     * passed pointer is a nullptr. If the pinned argument is true,
     * the new hostpotr is allocated via hipHostMalloc.
     * Otherwise, it is allocated via classic malloc.
     */
    __host__ hipError_t init(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int n1,int n2,
        int lb1,int lb2,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      this->data.wrap(
          data_host,
          data_dev,
          n1,n2,
          lb1,lb2
      );
      this->bytes_per_element = bytes_per_element;
      this->num_refs          = 1;
      this->alloc_mode        = alloc_mode;
      this->sync_mode         = sync_mode;
  
      // host array allocation
      hipError_t ierr = hipSuccess;
      switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
#endif
        case AllocMode::AllocPinnedHostWrapDevice:
        case AllocMode::AllocPinnedHostAllocDevice:
          ierr = hipHostMalloc((void**) &this->data.data_host,this->num_data_bytes(),0);
	  break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
          this->data.data_host = (T*) malloc(this->num_data_bytes());
	  break;
#endif
	case AllocMode::WrapHostWrapDevice:
        case AllocMode::WrapHostAllocDevice:
          // do nothing as already set by data.wrap call
	  break;
	default:
          std::cerr << "ERROR: gpufort::array2::init(...): Unexpected value for 'this->alloc_mode': " 
                    << static_cast<int>(this->alloc_mode) << std::endl; 
          std::terminate();
          break;
      }
      // device array allocation
      if ( ierr == hipSuccess ) {
        switch (this->alloc_mode) {
          case AllocMode::AllocHostWrapDevice:
          case AllocMode::AllocPinnedHostWrapDevice:
          case AllocMode::WrapHostWrapDevice:
            // do nothing as already set by data.wrap call
            break;
          case AllocMode::WrapHostAllocDevice:
          case AllocMode::AllocHostAllocDevice:
          case AllocMode::AllocPinnedHostAllocDevice:
            ierr = hipMalloc((void**) &this->data.data_dev, this->num_data_bytes());
            break;
          default:
            break;
        }
      }
      if ( ierr == hipSuccess ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::None:
          case SyncMode::Copyout:
          case SyncMode::InvertedCopyout:
            // do nothing
            break;
          case SyncMode::Copy:
          case SyncMode::Copyin:
            ierr = this->copy_to_device();
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyin:
            ierr = this->copy_to_host();
            break;
          default:
            std::cerr << "ERROR: gpufort::array2::init(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
      }
      return ierr;
    }
   
    __host__ hipError_t init(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int* sizes,
        int* lower_bounds,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      return this->init(
        bytes_per_element,
        data_host,
        data_dev,
        sizes[0],sizes[1],
        lower_bounds[0],lower_bounds[1],alloc_mode,sync_mode);
    }
    
    /**
     * Destroy associated host and device data
     * if this is not a wrapper. Depending
     * on the pinned attribue, hipHostFree 
     * is used instead of free to deallocate
     * the host data.
     */
    __host__ hipError_t destroy() {
      hipError_t ierr = hipSuccess;
      if ( bytes_per_element > 0 ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::Copy:
          case SyncMode::Copyout:
            ierr = this->copy_to_host();
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyout:
            ierr = this->copy_to_device();
            break;
          case SyncMode::None:
          case SyncMode::Copyin:
          case SyncMode::InvertedCopyin:
            // do nothing
            break;
          default:
            std::cerr << "ERROR: gpufort::array2::destroy(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
        // host array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
#endif
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipHostFree(this->data.data_host);
              break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
              free(this->data.data_host); 
              break;
#endif
            case AllocMode::WrapHostWrapDevice:
            case AllocMode::WrapHostAllocDevice:
              break;
            default:
              std::cerr << "ERROR: gpufort::array2::destroy(...): Unexpected value for 'alloc_mode': " 
                        << static_cast<int>(alloc_mode) << std::endl; 
              std::terminate();
              break;
          }
        }
        // device array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::WrapHostWrapDevice:
              // do nothing as already set by data.wrap call
              break;
            case AllocMode::WrapHostAllocDevice:
            case AllocMode::AllocHostAllocDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipFree(this->data.data_dev);
              break;
            default:
              break;
          }
        }
        if ( ierr == hipSuccess ) {
          this->data.data_host    = nullptr;
          this->data.data_dev     = nullptr;
          this->bytes_per_element = -1; 
        } 
      }
      return ierr;
    }
    
    /**
     * Copy data to a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the destinaton buffer is interpreted as host or device variable
     * and the source is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_to_buffer(
       void* buffer, hipMemcpyKind memcpy_kind
    ) {
      T* src = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyDeviceToHost: 
        case hipMemcpyDeviceToDevice:
          src = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyHostToDevice:
          src = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array2::copy_to_buffer(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(src!=nullptr);
      #endif
      return hipMemcpy(
        buffer, 
        (void*) src,
        this->num_data_bytes(), 
        memcpy_kind);
    }
    
    /**
     * Copy data from a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the source buffer is interpreted as host or device variable
     * and the destination is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_from_buffer(
       void* buffer, hipMemcpyKind memcpy_kind
    ) {
      T* dest = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyHostToDevice: 
        case hipMemcpyDeviceToDevice:
          dest = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyDeviceToHost:
          dest = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array2::copy_from_buffer(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(dest!=nullptr);
      #endif
      return hipMemcpy(
        (void*) dest,
        buffer, 
        this->num_data_bytes(), 
        memcpy_kind);
    }
    
    /**
     * Copy host data to the device.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_host() {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpy(
        (void*) this->data.data_host, 
        (void*) this->data.data_dev,
        this->num_data_bytes(), 
        hipMemcpyDeviceToHost);
    }
    
    /**
     * Copy device data to the host.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_device() {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpy(
        (void*) this->data.data_dev, 
        (void*) this->data.data_host,
        this->num_data_bytes(),
        hipMemcpyHostToDevice);
    }
    
    /** 
     * Copies the struct to the device into an pre-allocated
     * device memory block but does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[in] device_struct device memory address to copy to
     */
    __host__ hipError_t copy_self_to_device(
        gpufort::array2<T>* device_struct
    ) {
      const size_t size = sizeof(array2<char>); // sizeof(T*) = sizeof(char*)
      return hipMemcpy(
          (void*) device_struct, 
          (void*) this,
          size,
          hipMemcpyHostToDevice);
    }

    /** 
     * Copies the struct to the device but
     * does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[inout] device_copy pointer to device copy pointer 
     */
    __host__ hipError_t create_device_copy(
        gpufort::array2<T>** device_copy
    ) {
      const size_t size = sizeof(array2<char>); // sizeof(T*) = sizeof(char*)
      hipError_t ierr   = hipMalloc((void**)device_copy,size);      
      if ( ierr == hipSuccess ) {
        return this->copy_self_to_device(*device_copy);
      } else {
        return ierr;
      }
    }

    /**
     * Allocate a host buffer with
     * the same size as the data buffers
     * associated with this gpufort array.
     * @see num_data_bytes()
     * \param[inout] pointer to the buffer to allocate
     * \param[in] pinned If the memory should be pinned (default=True)
     * \param[in] flags  Flags for the host memory allocation (default=0).
     */
    __host__ hipError_t allocate_host_buffer(void** buffer,bool pinned=true,int flags=0) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
      if ( pinned ) { 
        ierr = hipHostMalloc(buffer,this->num_data_bytes(),flags);
      } else {
        *buffer = malloc(this->num_data_bytes());
      }
      return ierr;
    }

    /**
     * Deallocate a host buffer
     * created via the allocate_host_buffer routine.
     * @see num_data_bytes(), allocate_host_buffer
     * \param[inout] the buffer to deallocte
     * \param[in] pinned If the memory to deallocate is pinned (default=True)
     */
    __host__ hipError_t deallocate_host_buffer(void* buffer,bool pinned=true) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
      if ( pinned ) { 
        ierr = hipHostFree(buffer);
      } else {
        free(buffer);
      }
      return ierr;
    }
    /**
     * Allocate a device buffer with
     * the same size as the data buffers
     * associated with this gpufort array.
     * @see num_data_bytes()
     * \param[inout] pointer to the buffer to allocate
     */
    __host__ hipError_t allocate_device_buffer(void** buffer) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
 
      ierr = hipMalloc(buffer,this->num_data_bytes());
      return ierr;
    }

    /**
     * Deallocate a device buffer
     * created via the allocate_device_buffer routine.
     * @see num_data_bytes(), allocate_device_buffer
     * \param[inout] the buffer to deallocte
     */
    __host__ hipError_t deallocate_device_buffer(void* buffer) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
 
      ierr = hipFree(buffer);
      return ierr;
    }

    /** 
     * Copy metadata from another gpufort array to this gpufort array.
     * \note The allocation mode is always set to AllocMode::WrapHostWrapDevice
     *       and the sync mode is set to SyncMode::None.
     * \note No deep copy, i.e. the host and device data is not copied, just the pointers
     * to the data.
     */
    __host__ void copy(const gpufort::array2<T>& other) {
      this->init(
        other.bytes_per_element,
        other.data.data_host,
        other.data.data_dev,
        other.size(1),other.size(2),
        other.lbound(1),other.lbound(2),
        AllocMode::WrapHostWrapDevice,
        SyncMode::None);
    }

    /** 
     *  \return Number of bytes required to store 
     *  the array elements on the host or the device.
     */ 
    __host__ size_t num_data_bytes() {
      return this->data.num_elements * this->bytes_per_element;
    }
    
    /**
     * \return the number of elements of this array.
     */
    __host__ int num_elements() {
      return this->data.num_elements;
    }

    /**
     * Linearize multi-dimensional index.
     *
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ int linearized_index (
      const int i1,
      const int i2
    ) const {
      return this->data.linearized_index(
        i1,i2
      );
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ T& operator() (
      const int i1,
      const int i2
    ) {
      return this->data(
        i1,i2
      );
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ const T& operator() (
      const int i1,
      const int i2
    ) const {
      return this->data(
        i1,i2
      );
    }
    
    /**
     * \return Size of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,2
     */
    __host__ __device__ __forceinline__ int size(int dim) const {
      return this->data.size(dim);
    }
    
    /**
     * \return Lower bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,2
     */
    __host__ __device__ __forceinline__ int lbound(int dim) const {
      return this->data.lbound(dim);
    }
    
    /**
     * \return Upper bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,2
     */
    __host__ __device__ __forceinline__ int ubound(int dim) const {
      return this->data.ubound(dim);
    }
    

    /**
     * Collapse the array by fixing 1 indices.
     * \return A gpufort array of rank 1.
     * \param[in] i2,...,i2 indices to fix.
     */
    __host__ gpufort::array1<T> collapse(
        const int i2
    ) const {
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1);
      const int index = linearized_index(
        lb1,
        i2);
      T* data_host_new = nullptr; 
      if (this->data.data_host != nullptr) {
        data_host_new = this->data.data_host + index;
      }
      T* data_dev_new = nullptr; 
      if (this->data.data_dev != nullptr) {
        data_dev_new = this->data.data_dev + index;
      }
      gpufort::array1<T> collapsed_array;
      hipError_t ierr = 
        collapsed_array.init(
          this->bytes_per_element,
          data_host_new,
          data_dev_new,
          n1,
          lb1,
          AllocMode::WrapHostWrapDevice,
          SyncMode::None);
      return collapsed_array;
    }

    /**
     * Write host data values and their corresponding index
     * to an output stream.
     * \param[inout] out        output stream
     * \param[in]    prefix     prefix to put before each output line
     * \param[in]    print_prec precision to use when printing floating point numbers 
     *               [default=GPUFORT_ARRAY_PRINT_PREC[default=6]]
     * \note This method only makes sense for numeric data types and
     *       may result in compilation, runtime errors or undefined output if the
     *       underlying data is not numeric.
     */
    __host__ void print_host_data(
        std::ostream& out, const char* prefix, PrintMode print_mode,
        const int print_prec=GPUFORT_ARRAY_PRINT_PREC) const {
      const bool print_norms = 
	print_mode==PrintMode::PrintNorms ||
        print_mode==PrintMode::PrintValuesAndNorms ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      const bool print_values = 
        print_mode==PrintMode::PrintValues ||
        print_mode==PrintMode::PrintValuesAndNorms;
      const bool print_values_c_style = 
        print_mode==PrintMode::PrintCValues ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      if ( !print_norms && !print_values && !print_values_c_style ) {
        std::cerr << "ERROR: gpufort::array2::print_host_data(...): Unexpected value for 'print_mode': " 
                  << static_cast<int>(print_mode) << std::endl; 
        std::terminate();
      }
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1); 
      const int n2  = this->size(2);
      const int lb2 = this->lbound(2); 
      const int n = this->data.num_elements; 
      T* A_h   = this->data.data_host;
      T min = +std::numeric_limits<T>::max();
      T max = -std::numeric_limits<T>::max();
      T sum = 0;
      T l1  = 0;
      T l2  = 0;
      out << prefix << ":\n";
    for ( int i2 = 0; i2 < n2; i2++ ) {
    for ( int i1 = 0; i1 < n1; i1++ ) {
      const int idx = this->linearized_index(i1,i2);
      T value = A_h[idx];
      if ( print_norms ) {
        min  = std::min(value,min);
        max  = std::max(value,max);
        sum += value;
        l1  += std::abs(value);
        l2  += value*value;
      }
      if ( print_values ) {
          out << prefix << "(" << (lb1+i1) << "," << (lb2+i2) <<  ") = " << std::setprecision(print_prec) << value << "\n";
      } else if ( print_values_c_style ) {
          out << prefix << "[" << idx << "] = " << std::setprecision(print_prec) << value << "\n";
      }
    }} // for loops
    if ( print_norms ) {
      out << prefix << ":min=" << std::setprecision(print_prec) << min << "\n";
      out << prefix << ":max=" << std::setprecision(print_prec) << max << "\n";
      out << prefix << ":sum=" << std::setprecision(print_prec) << sum << "\n";
      out << prefix << ":l1="  << std::setprecision(print_prec) << l1  << "\n";
      out << prefix << ":l2="  << std::setprecision(print_prec) << std::sqrt(l2) << "\n";
      out << prefix << "num_elements=" << std::setprecision(print_prec) << n << "\n";
    }
  }
    /**
     * Write device data values and their corresponding index
     * to an output stream.
     * \param[inout] out        output stream
     * \param[in]    prefix     prefix to put before each output line
     * \param[in]    print_prec precision to use when printing floating point numbers 
     *               [default=GPUFORT_ARRAY_PRINT_PREC[default=6]]
     * \note This method only makes sense for numeric data types and
     *       may result in compilation, runtime errors or undefined output if the
     *       underlying data is not numeric.
     */
    __host__ void print_device_data(
        std::ostream& out, const char* prefix, PrintMode print_mode,
        const int print_prec=GPUFORT_ARRAY_PRINT_PREC) const {
      const bool print_norms = 
	print_mode==PrintMode::PrintNorms ||
        print_mode==PrintMode::PrintValuesAndNorms ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      const bool print_values = 
        print_mode==PrintMode::PrintValues ||
        print_mode==PrintMode::PrintValuesAndNorms;
      const bool print_values_c_style = 
        print_mode==PrintMode::PrintCValues ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      if ( !print_norms && !print_values && !print_values_c_style ) {
        std::cerr << "ERROR: gpufort::array2::print_device_data(...): Unexpected value for 'print_mode': " 
                  << static_cast<int>(print_mode) << std::endl; 
        std::terminate();
      }
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1); 
      const int n2  = this->size(2);
      const int lb2 = this->lbound(2); 
      const int n = this->data.num_elements; 
      std::vector<T> host_array(n);
      T* A_h = host_array.data();
      T* A   = this->data.data_dev;
      HIP_CHECK(hipMemcpy(A_h, A, n*sizeof(T), hipMemcpyDeviceToHost));
      T min = +std::numeric_limits<T>::max();
      T max = -std::numeric_limits<T>::max();
      T sum = 0;
      T l1  = 0;
      T l2  = 0;
      out << prefix << ":\n";
    for ( int i2 = 0; i2 < n2; i2++ ) {
    for ( int i1 = 0; i1 < n1; i1++ ) {
      const int idx = this->linearized_index(i1,i2);
      T value = A_h[idx];
      if ( print_norms ) {
        min  = std::min(value,min);
        max  = std::max(value,max);
        sum += value;
        l1  += std::abs(value);
        l2  += value*value;
      }
      if ( print_values ) {
          out << prefix << "(" << (lb1+i1) << "," << (lb2+i2) <<  ") = " << std::setprecision(print_prec) << value << "\n";
      } else if ( print_values_c_style ) {
          out << prefix << "[" << idx << "] = " << std::setprecision(print_prec) << value << "\n";
      }
    }} // for loops
    if ( print_norms ) {
      out << prefix << ":min=" << std::setprecision(print_prec) << min << "\n";
      out << prefix << ":max=" << std::setprecision(print_prec) << max << "\n";
      out << prefix << ":sum=" << std::setprecision(print_prec) << sum << "\n";
      out << prefix << ":l1="  << std::setprecision(print_prec) << l1  << "\n";
      out << prefix << ":l2="  << std::setprecision(print_prec) << std::sqrt(l2) << "\n";
      out << prefix << "num_elements=" << std::setprecision(print_prec) << n << "\n";
    }
  }

  };

  /**
   * Intended to be passed as kernel launch parameter.
   * \note Operator `T& operator()` requires parametrization with type T.
   * \note Size of this struct independent of template type T.
   */
  template<typename T>
  struct array_descr3 {
    T*     data_host = nullptr;
    T*     data_dev  = nullptr;
    int num_elements = 0;     //> Number of elements represented by this array.
    int index_offset = -1;    //> Offset for index calculation; scalar product of negative lower bounds and strides.
    int    stride2  = -1;    //> Stride 2 for linearizing 3-dimensional index.
    int    stride3  = -1;    //> Stride 3 for linearizing 3-dimensional index.
 
    /**
     * Wrap a host-device pointer pair and store the associated
     * array meta data.
     *
     * \param[in] data_host host data pointer.
     * \param[in] data_dev device data pointer.
     * \param[in] n1,n2,... element counts per dimension of the multi-dim. array 
     * \param[in] lb1,lb2,... lower bounds, i.e. the index of the first element per dimension of the multi-dim. array 
     */ 
    __host__ __device__ void wrap(
        T* data_host,
        T* data_dev,
        const int n1,
        const int n2,
        const int n3,
        const int lb1,
        const int lb2,
        const int lb3
      ) {
       this->data_host = data_host;
       this->data_dev  = data_dev;
       // column-major access
       this->stride2 = 1*n1;
       this->stride3 = this->stride2*n2;
       this->num_elements = this->stride3*n3;
       this->index_offset =
         -lb1*1
         -lb2*this->stride2
         -lb3*this->stride3;
    }
    
    /**
     * Linearize multi-dimensional index.
     *
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ int linearized_index (
      const int i1,
      const int i2,
      const int i3
    ) const {
      return this->index_offset
          + i1*1
          + i2*this->stride2
          + i3*this->stride3;
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ T& operator() (
      const int i1,
      const int i2,
      const int i3
    ) {
      const int index = linearized_index(
        i1,i2,i3
      );
      #ifdef __HIP_DEVICE_COMPILE__
      return this->data_dev[index];
      #else
      assert(i1 >= lbound(1));
      assert(i1 <= ubound(1));
      assert(i2 >= lbound(2));
      assert(i2 <= ubound(2));
      assert(i3 >= lbound(3));
      assert(i3 <= ubound(3));
      return this->data_host[index];
      #endif
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ const T& operator() (
      const int i1,
      const int i2,
      const int i3
    ) const {
      const int index = linearized_index(
        i1,i2,i3
      );
      #ifdef __HIP_DEVICE_COMPILE__
      return this->data_dev[index];
      #else
      assert(i1 >= lbound(1));
      assert(i1 <= ubound(1));
      assert(i2 >= lbound(2));
      assert(i2 <= ubound(2));
      assert(i3 >= lbound(3));
      assert(i3 <= ubound(3));
      return this->data_host[index];
      #endif
    }
    
    /**
     * \return Size of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,3
     */
    __host__ __device__ __forceinline__ int size(int dim) const {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(dim >= 1);
      assert(dim <= 3);
      #endif
      switch(dim) {
       case 3:
	        return this->num_elements / this->stride3;
       case 2:
	        return this->stride3 / this->stride2;
       case 1:
	        return this->stride2 / 1;
       default:
          #ifndef __HIP_DEVICE_COMPILE__
          std::cerr << "‘dim’ argument of ‘gpufort::array3::size’ is not a valid dimension index ('dim': "<<dim<<", max dimension: 3" << std::endl;
          std::terminate();
          #else
          return -1;
          #endif
      }
    }
    
    /**
     * \return Lower bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,3
     */
    __host__ __device__ __forceinline__ int lbound(int dim) const {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(dim >= 1);
      assert(dim <= 3);
      #endif
      int offset_remainder = this->index_offset;
      int lb = -offset_remainder / this->stride3;
      if ( dim == 3 ) return lb;
      offset_remainder = offset_remainder + lb*this->stride3;
      lb = -offset_remainder / this->stride2;
      if ( dim == 2 ) return lb;
      offset_remainder = offset_remainder + lb*this->stride2;
      lb = -offset_remainder / 1;
      #ifndef __HIP_DEVICE_COMPILE__
      offset_remainder = offset_remainder + lb*1;
      assert(offset_remainder == 0);
      #endif
      return lb;
    }
    
    /**
     * \return Upper bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,3
     */
    __host__ __device__ __forceinline__ int ubound(int dim) const {
      return this->lbound(dim) + this->size(dim) - 1;
    }
  };

  template<typename T>
  struct array3{
    array_descr3<T> data;
    AllocMode              alloc_mode = AllocMode::WrapHostAllocDevice; //> Data allocation strategy. Default: 
                                                                        //> wrap the host and allocate device data
    SyncMode               sync_mode  = SyncMode::None;                 //> How data should be synchronized
                                                                        //> during the initialization and destruction of this GPUFORT array.
    int num_refs             = 0;  //> Number of references.
    size_t bytes_per_element = -1; //> Bytes per element; stored to make num_data_bytes routine independent of T 

    array3() {
      // do nothing
    }
    
    ~array3() {
      // do nothing
    }

    /**
     * Initialize.
     * \param[in] data_host host data pointer (may be nullptr; see the note).
     * \param[in] pinned the host array is pinned or should be pinned.
     * \param[in] copyout_at_destruction copy data out from the
     *            device to the host when this instance gets destroyed.
     * \note Allocates a new host_data pointer if the
     * passed pointer is a nullptr. If the pinned argument is true,
     * the new hostpotr is allocated via hipHostMalloc.
     * Otherwise, it is allocated via classic malloc.
     */
    __host__ hipError_t init_async(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int n1,int n2,int n3,
        int lb1,int lb2,int lb3,
        hipStream_t stream,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      this->data.wrap(
          data_host,
          data_dev,
          n1,n2,n3,
          lb1,lb2,lb3
      );
      this->bytes_per_element = bytes_per_element;
      this->num_refs          = 1;
      this->alloc_mode        = alloc_mode;
      this->sync_mode         = sync_mode;
  
      // host array allocation
      hipError_t ierr = hipSuccess;
      switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
#endif
        case AllocMode::AllocPinnedHostWrapDevice:
        case AllocMode::AllocPinnedHostAllocDevice:
          ierr = hipHostMalloc((void**) &this->data.data_host,this->num_data_bytes(),0);
	  break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
          this->data.data_host = (T*) malloc(this->num_data_bytes());
	  break;
#endif
	case AllocMode::WrapHostWrapDevice:
        case AllocMode::WrapHostAllocDevice:
          // do nothing as already set by data.wrap call
	  break;
	default:
          std::cerr << "ERROR: gpufort::array3::init_async(...): Unexpected value for 'this->alloc_mode': " 
                    << static_cast<int>(this->alloc_mode) << std::endl; 
          std::terminate();
          break;
      }
      // device array allocation
      if ( ierr == hipSuccess ) {
        switch (this->alloc_mode) {
          case AllocMode::AllocHostWrapDevice:
          case AllocMode::AllocPinnedHostWrapDevice:
          case AllocMode::WrapHostWrapDevice:
            // do nothing as already set by data.wrap call
            break;
          case AllocMode::WrapHostAllocDevice:
          case AllocMode::AllocHostAllocDevice:
          case AllocMode::AllocPinnedHostAllocDevice:
            ierr = hipMalloc((void**) &this->data.data_dev, this->num_data_bytes());
            break;
          default:
            break;
        }
      }
      if ( ierr == hipSuccess ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::None:
          case SyncMode::Copyout:
          case SyncMode::InvertedCopyout:
            // do nothing
            break;
          case SyncMode::Copy:
          case SyncMode::Copyin:
            ierr = this->copy_to_device_async(stream);
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyin:
            ierr = this->copy_to_host_async(stream);
            break;
          default:
            std::cerr << "ERROR: gpufort::array3::init_async(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
      }
      return ierr;
    }
   
    __host__ hipError_t init_async(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int* sizes,
        int* lower_bounds,
        hipStream_t stream,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      return this->init_async(
        bytes_per_element,
        data_host,
        data_dev,
        sizes[0],sizes[1],sizes[2],
        lower_bounds[0],lower_bounds[1],lower_bounds[2],
        stream,alloc_mode,sync_mode);
    }
    
    /**
     * Destroy associated host and device data
     * if this is not a wrapper. Depending
     * on the pinned attribue, hipHostFree 
     * is used instead of free to deallocate
     * the host data.
     */
    __host__ hipError_t destroy_async(hipStream_t stream) {
      hipError_t ierr = hipSuccess;
      if ( bytes_per_element > 0 ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::Copy:
          case SyncMode::Copyout:
            ierr = this->copy_to_host_async(stream);
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyout:
            ierr = this->copy_to_device_async(stream);
            break;
          case SyncMode::None:
          case SyncMode::Copyin:
          case SyncMode::InvertedCopyin:
            // do nothing
            break;
          default:
            std::cerr << "ERROR: gpufort::array3::destroy_async(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
        // host array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
#endif
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipHostFree(this->data.data_host);
              break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
              free(this->data.data_host); 
              break;
#endif
            case AllocMode::WrapHostWrapDevice:
            case AllocMode::WrapHostAllocDevice:
              break;
            default:
              std::cerr << "ERROR: gpufort::array3::destroy_async(...): Unexpected value for 'alloc_mode': " 
                        << static_cast<int>(alloc_mode) << std::endl; 
              std::terminate();
              break;
          }
        }
        // device array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::WrapHostWrapDevice:
              // do nothing as already set by data.wrap call
              break;
            case AllocMode::WrapHostAllocDevice:
            case AllocMode::AllocHostAllocDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipFree(this->data.data_dev);
              break;
            default:
              break;
          }
        }
        if ( ierr == hipSuccess ) {
          this->data.data_host    = nullptr;
          this->data.data_dev     = nullptr;
          this->bytes_per_element = -1; 
        } 
      }
      return ierr;
    }
    
    /**
     * Copy data to a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the destinaton buffer is interpreted as host or device variable
     * and the source is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_to_buffer_async(
       void* buffer, hipMemcpyKind memcpy_kind,
       hipStream_t stream
    ) {
      T* src = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyDeviceToHost: 
        case hipMemcpyDeviceToDevice:
          src = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyHostToDevice:
          src = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array3::copy_to_buffer_async(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(src!=nullptr);
      #endif
      return hipMemcpyAsync(
        buffer, 
        (void*) src,
        this->num_data_bytes(), 
        memcpy_kind, stream);
    }
    
    /**
     * Copy data from a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the source buffer is interpreted as host or device variable
     * and the destination is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_from_buffer_async(
       void* buffer, hipMemcpyKind memcpy_kind,
       hipStream_t stream
    ) {
      T* dest = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyHostToDevice: 
        case hipMemcpyDeviceToDevice:
          dest = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyDeviceToHost:
          dest = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array3::copy_from_buffer_async(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(dest!=nullptr);
      #endif
      return hipMemcpyAsync(
        (void*) dest,
        buffer, 
        this->num_data_bytes(), 
        memcpy_kind, stream);
    }
    
    /**
     * Copy host data to the device.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_host_async(hipStream_t stream) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpyAsync(
        (void*) this->data.data_host, 
        (void*) this->data.data_dev,
        this->num_data_bytes(), 
        hipMemcpyDeviceToHost, stream);
    }
    
    /**
     * Copy device data to the host.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_device_async(hipStream_t stream) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpyAsync(
        (void*) this->data.data_dev, 
        (void*) this->data.data_host,
        this->num_data_bytes(),
        hipMemcpyHostToDevice, stream);
    }
    
    /** 
     * Copies the struct to the device into an pre-allocated
     * device memory block but does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[in] device_struct device memory address to copy to
     */
    __host__ hipError_t copy_self_to_device_async(
        gpufort::array3<T>* device_struct,
        hipStream_t stream
    ) {
      const size_t size = sizeof(array3<char>); // sizeof(T*) = sizeof(char*)
      return hipMemcpyAsync(
          (void*) device_struct, 
          (void*) this,
          size,
          hipMemcpyHostToDevice, stream);
    }

    /** 
     * Copies the struct to the device but
     * does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[inout] device_copy pointer to device copy pointer 
     */
    __host__ hipError_t create_device_copy_async(
        gpufort::array3<T>** device_copy,
        hipStream_t stream
    ) {
      const size_t size = sizeof(array3<char>); // sizeof(T*) = sizeof(char*)
      hipError_t ierr   = hipMalloc((void**)device_copy,size);      
      if ( ierr == hipSuccess ) {
        return this->copy_self_to_device_async(*device_copy,stream);
      } else {
        return ierr;
      }
    }
    /**
     * Initialize.
     * \param[in] data_host host data pointer (may be nullptr; see the note).
     * \param[in] pinned the host array is pinned or should be pinned.
     * \param[in] copyout_at_destruction copy data out from the
     *            device to the host when this instance gets destroyed.
     * \note Allocates a new host_data pointer if the
     * passed pointer is a nullptr. If the pinned argument is true,
     * the new hostpotr is allocated via hipHostMalloc.
     * Otherwise, it is allocated via classic malloc.
     */
    __host__ hipError_t init(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int n1,int n2,int n3,
        int lb1,int lb2,int lb3,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      this->data.wrap(
          data_host,
          data_dev,
          n1,n2,n3,
          lb1,lb2,lb3
      );
      this->bytes_per_element = bytes_per_element;
      this->num_refs          = 1;
      this->alloc_mode        = alloc_mode;
      this->sync_mode         = sync_mode;
  
      // host array allocation
      hipError_t ierr = hipSuccess;
      switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
#endif
        case AllocMode::AllocPinnedHostWrapDevice:
        case AllocMode::AllocPinnedHostAllocDevice:
          ierr = hipHostMalloc((void**) &this->data.data_host,this->num_data_bytes(),0);
	  break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
          this->data.data_host = (T*) malloc(this->num_data_bytes());
	  break;
#endif
	case AllocMode::WrapHostWrapDevice:
        case AllocMode::WrapHostAllocDevice:
          // do nothing as already set by data.wrap call
	  break;
	default:
          std::cerr << "ERROR: gpufort::array3::init(...): Unexpected value for 'this->alloc_mode': " 
                    << static_cast<int>(this->alloc_mode) << std::endl; 
          std::terminate();
          break;
      }
      // device array allocation
      if ( ierr == hipSuccess ) {
        switch (this->alloc_mode) {
          case AllocMode::AllocHostWrapDevice:
          case AllocMode::AllocPinnedHostWrapDevice:
          case AllocMode::WrapHostWrapDevice:
            // do nothing as already set by data.wrap call
            break;
          case AllocMode::WrapHostAllocDevice:
          case AllocMode::AllocHostAllocDevice:
          case AllocMode::AllocPinnedHostAllocDevice:
            ierr = hipMalloc((void**) &this->data.data_dev, this->num_data_bytes());
            break;
          default:
            break;
        }
      }
      if ( ierr == hipSuccess ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::None:
          case SyncMode::Copyout:
          case SyncMode::InvertedCopyout:
            // do nothing
            break;
          case SyncMode::Copy:
          case SyncMode::Copyin:
            ierr = this->copy_to_device();
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyin:
            ierr = this->copy_to_host();
            break;
          default:
            std::cerr << "ERROR: gpufort::array3::init(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
      }
      return ierr;
    }
   
    __host__ hipError_t init(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int* sizes,
        int* lower_bounds,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      return this->init(
        bytes_per_element,
        data_host,
        data_dev,
        sizes[0],sizes[1],sizes[2],
        lower_bounds[0],lower_bounds[1],lower_bounds[2],alloc_mode,sync_mode);
    }
    
    /**
     * Destroy associated host and device data
     * if this is not a wrapper. Depending
     * on the pinned attribue, hipHostFree 
     * is used instead of free to deallocate
     * the host data.
     */
    __host__ hipError_t destroy() {
      hipError_t ierr = hipSuccess;
      if ( bytes_per_element > 0 ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::Copy:
          case SyncMode::Copyout:
            ierr = this->copy_to_host();
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyout:
            ierr = this->copy_to_device();
            break;
          case SyncMode::None:
          case SyncMode::Copyin:
          case SyncMode::InvertedCopyin:
            // do nothing
            break;
          default:
            std::cerr << "ERROR: gpufort::array3::destroy(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
        // host array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
#endif
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipHostFree(this->data.data_host);
              break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
              free(this->data.data_host); 
              break;
#endif
            case AllocMode::WrapHostWrapDevice:
            case AllocMode::WrapHostAllocDevice:
              break;
            default:
              std::cerr << "ERROR: gpufort::array3::destroy(...): Unexpected value for 'alloc_mode': " 
                        << static_cast<int>(alloc_mode) << std::endl; 
              std::terminate();
              break;
          }
        }
        // device array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::WrapHostWrapDevice:
              // do nothing as already set by data.wrap call
              break;
            case AllocMode::WrapHostAllocDevice:
            case AllocMode::AllocHostAllocDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipFree(this->data.data_dev);
              break;
            default:
              break;
          }
        }
        if ( ierr == hipSuccess ) {
          this->data.data_host    = nullptr;
          this->data.data_dev     = nullptr;
          this->bytes_per_element = -1; 
        } 
      }
      return ierr;
    }
    
    /**
     * Copy data to a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the destinaton buffer is interpreted as host or device variable
     * and the source is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_to_buffer(
       void* buffer, hipMemcpyKind memcpy_kind
    ) {
      T* src = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyDeviceToHost: 
        case hipMemcpyDeviceToDevice:
          src = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyHostToDevice:
          src = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array3::copy_to_buffer(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(src!=nullptr);
      #endif
      return hipMemcpy(
        buffer, 
        (void*) src,
        this->num_data_bytes(), 
        memcpy_kind);
    }
    
    /**
     * Copy data from a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the source buffer is interpreted as host or device variable
     * and the destination is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_from_buffer(
       void* buffer, hipMemcpyKind memcpy_kind
    ) {
      T* dest = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyHostToDevice: 
        case hipMemcpyDeviceToDevice:
          dest = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyDeviceToHost:
          dest = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array3::copy_from_buffer(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(dest!=nullptr);
      #endif
      return hipMemcpy(
        (void*) dest,
        buffer, 
        this->num_data_bytes(), 
        memcpy_kind);
    }
    
    /**
     * Copy host data to the device.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_host() {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpy(
        (void*) this->data.data_host, 
        (void*) this->data.data_dev,
        this->num_data_bytes(), 
        hipMemcpyDeviceToHost);
    }
    
    /**
     * Copy device data to the host.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_device() {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpy(
        (void*) this->data.data_dev, 
        (void*) this->data.data_host,
        this->num_data_bytes(),
        hipMemcpyHostToDevice);
    }
    
    /** 
     * Copies the struct to the device into an pre-allocated
     * device memory block but does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[in] device_struct device memory address to copy to
     */
    __host__ hipError_t copy_self_to_device(
        gpufort::array3<T>* device_struct
    ) {
      const size_t size = sizeof(array3<char>); // sizeof(T*) = sizeof(char*)
      return hipMemcpy(
          (void*) device_struct, 
          (void*) this,
          size,
          hipMemcpyHostToDevice);
    }

    /** 
     * Copies the struct to the device but
     * does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[inout] device_copy pointer to device copy pointer 
     */
    __host__ hipError_t create_device_copy(
        gpufort::array3<T>** device_copy
    ) {
      const size_t size = sizeof(array3<char>); // sizeof(T*) = sizeof(char*)
      hipError_t ierr   = hipMalloc((void**)device_copy,size);      
      if ( ierr == hipSuccess ) {
        return this->copy_self_to_device(*device_copy);
      } else {
        return ierr;
      }
    }

    /**
     * Allocate a host buffer with
     * the same size as the data buffers
     * associated with this gpufort array.
     * @see num_data_bytes()
     * \param[inout] pointer to the buffer to allocate
     * \param[in] pinned If the memory should be pinned (default=True)
     * \param[in] flags  Flags for the host memory allocation (default=0).
     */
    __host__ hipError_t allocate_host_buffer(void** buffer,bool pinned=true,int flags=0) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
      if ( pinned ) { 
        ierr = hipHostMalloc(buffer,this->num_data_bytes(),flags);
      } else {
        *buffer = malloc(this->num_data_bytes());
      }
      return ierr;
    }

    /**
     * Deallocate a host buffer
     * created via the allocate_host_buffer routine.
     * @see num_data_bytes(), allocate_host_buffer
     * \param[inout] the buffer to deallocte
     * \param[in] pinned If the memory to deallocate is pinned (default=True)
     */
    __host__ hipError_t deallocate_host_buffer(void* buffer,bool pinned=true) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
      if ( pinned ) { 
        ierr = hipHostFree(buffer);
      } else {
        free(buffer);
      }
      return ierr;
    }
    /**
     * Allocate a device buffer with
     * the same size as the data buffers
     * associated with this gpufort array.
     * @see num_data_bytes()
     * \param[inout] pointer to the buffer to allocate
     */
    __host__ hipError_t allocate_device_buffer(void** buffer) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
 
      ierr = hipMalloc(buffer,this->num_data_bytes());
      return ierr;
    }

    /**
     * Deallocate a device buffer
     * created via the allocate_device_buffer routine.
     * @see num_data_bytes(), allocate_device_buffer
     * \param[inout] the buffer to deallocte
     */
    __host__ hipError_t deallocate_device_buffer(void* buffer) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
 
      ierr = hipFree(buffer);
      return ierr;
    }

    /** 
     * Copy metadata from another gpufort array to this gpufort array.
     * \note The allocation mode is always set to AllocMode::WrapHostWrapDevice
     *       and the sync mode is set to SyncMode::None.
     * \note No deep copy, i.e. the host and device data is not copied, just the pointers
     * to the data.
     */
    __host__ void copy(const gpufort::array3<T>& other) {
      this->init(
        other.bytes_per_element,
        other.data.data_host,
        other.data.data_dev,
        other.size(1),other.size(2),other.size(3),
        other.lbound(1),other.lbound(2),other.lbound(3),
        AllocMode::WrapHostWrapDevice,
        SyncMode::None);
    }

    /** 
     *  \return Number of bytes required to store 
     *  the array elements on the host or the device.
     */ 
    __host__ size_t num_data_bytes() {
      return this->data.num_elements * this->bytes_per_element;
    }
    
    /**
     * \return the number of elements of this array.
     */
    __host__ int num_elements() {
      return this->data.num_elements;
    }

    /**
     * Linearize multi-dimensional index.
     *
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ int linearized_index (
      const int i1,
      const int i2,
      const int i3
    ) const {
      return this->data.linearized_index(
        i1,i2,i3
      );
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ T& operator() (
      const int i1,
      const int i2,
      const int i3
    ) {
      return this->data(
        i1,i2,i3
      );
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ const T& operator() (
      const int i1,
      const int i2,
      const int i3
    ) const {
      return this->data(
        i1,i2,i3
      );
    }
    
    /**
     * \return Size of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,3
     */
    __host__ __device__ __forceinline__ int size(int dim) const {
      return this->data.size(dim);
    }
    
    /**
     * \return Lower bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,3
     */
    __host__ __device__ __forceinline__ int lbound(int dim) const {
      return this->data.lbound(dim);
    }
    
    /**
     * \return Upper bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,3
     */
    __host__ __device__ __forceinline__ int ubound(int dim) const {
      return this->data.ubound(dim);
    }
    

    /**
     * Collapse the array by fixing 1 indices.
     * \return A gpufort array of rank 2.
     * \param[in] i3,...,i3 indices to fix.
     */
    __host__ gpufort::array2<T> collapse(
        const int i3
    ) const {
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1);
      const int n2  = this->size(2);
      const int lb2 = this->lbound(2);
      const int index = linearized_index(
        lb1,lb2,
        i3);
      T* data_host_new = nullptr; 
      if (this->data.data_host != nullptr) {
        data_host_new = this->data.data_host + index;
      }
      T* data_dev_new = nullptr; 
      if (this->data.data_dev != nullptr) {
        data_dev_new = this->data.data_dev + index;
      }
      gpufort::array2<T> collapsed_array;
      hipError_t ierr = 
        collapsed_array.init(
          this->bytes_per_element,
          data_host_new,
          data_dev_new,
          n1,n2,
          lb1,lb2,
          AllocMode::WrapHostWrapDevice,
          SyncMode::None);
      return collapsed_array;
    }
    /**
     * Collapse the array by fixing 2 indices.
     * \return A gpufort array of rank 1.
     * \param[in] i2,...,i3 indices to fix.
     */
    __host__ gpufort::array1<T> collapse(
        const int i2,
        const int i3
    ) const {
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1);
      const int index = linearized_index(
        lb1,
        i2,i3);
      T* data_host_new = nullptr; 
      if (this->data.data_host != nullptr) {
        data_host_new = this->data.data_host + index;
      }
      T* data_dev_new = nullptr; 
      if (this->data.data_dev != nullptr) {
        data_dev_new = this->data.data_dev + index;
      }
      gpufort::array1<T> collapsed_array;
      hipError_t ierr = 
        collapsed_array.init(
          this->bytes_per_element,
          data_host_new,
          data_dev_new,
          n1,
          lb1,
          AllocMode::WrapHostWrapDevice,
          SyncMode::None);
      return collapsed_array;
    }

    /**
     * Write host data values and their corresponding index
     * to an output stream.
     * \param[inout] out        output stream
     * \param[in]    prefix     prefix to put before each output line
     * \param[in]    print_prec precision to use when printing floating point numbers 
     *               [default=GPUFORT_ARRAY_PRINT_PREC[default=6]]
     * \note This method only makes sense for numeric data types and
     *       may result in compilation, runtime errors or undefined output if the
     *       underlying data is not numeric.
     */
    __host__ void print_host_data(
        std::ostream& out, const char* prefix, PrintMode print_mode,
        const int print_prec=GPUFORT_ARRAY_PRINT_PREC) const {
      const bool print_norms = 
	print_mode==PrintMode::PrintNorms ||
        print_mode==PrintMode::PrintValuesAndNorms ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      const bool print_values = 
        print_mode==PrintMode::PrintValues ||
        print_mode==PrintMode::PrintValuesAndNorms;
      const bool print_values_c_style = 
        print_mode==PrintMode::PrintCValues ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      if ( !print_norms && !print_values && !print_values_c_style ) {
        std::cerr << "ERROR: gpufort::array3::print_host_data(...): Unexpected value for 'print_mode': " 
                  << static_cast<int>(print_mode) << std::endl; 
        std::terminate();
      }
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1); 
      const int n2  = this->size(2);
      const int lb2 = this->lbound(2); 
      const int n3  = this->size(3);
      const int lb3 = this->lbound(3); 
      const int n = this->data.num_elements; 
      T* A_h   = this->data.data_host;
      T min = +std::numeric_limits<T>::max();
      T max = -std::numeric_limits<T>::max();
      T sum = 0;
      T l1  = 0;
      T l2  = 0;
      out << prefix << ":\n";
    for ( int i3 = 0; i3 < n3; i3++ ) {
    for ( int i2 = 0; i2 < n2; i2++ ) {
    for ( int i1 = 0; i1 < n1; i1++ ) {
      const int idx = this->linearized_index(i1,i2,i3);
      T value = A_h[idx];
      if ( print_norms ) {
        min  = std::min(value,min);
        max  = std::max(value,max);
        sum += value;
        l1  += std::abs(value);
        l2  += value*value;
      }
      if ( print_values ) {
          out << prefix << "(" << (lb1+i1) << "," << (lb2+i2) << "," << (lb3+i3) <<  ") = " << std::setprecision(print_prec) << value << "\n";
      } else if ( print_values_c_style ) {
          out << prefix << "[" << idx << "] = " << std::setprecision(print_prec) << value << "\n";
      }
    }}} // for loops
    if ( print_norms ) {
      out << prefix << ":min=" << std::setprecision(print_prec) << min << "\n";
      out << prefix << ":max=" << std::setprecision(print_prec) << max << "\n";
      out << prefix << ":sum=" << std::setprecision(print_prec) << sum << "\n";
      out << prefix << ":l1="  << std::setprecision(print_prec) << l1  << "\n";
      out << prefix << ":l2="  << std::setprecision(print_prec) << std::sqrt(l2) << "\n";
      out << prefix << "num_elements=" << std::setprecision(print_prec) << n << "\n";
    }
  }
    /**
     * Write device data values and their corresponding index
     * to an output stream.
     * \param[inout] out        output stream
     * \param[in]    prefix     prefix to put before each output line
     * \param[in]    print_prec precision to use when printing floating point numbers 
     *               [default=GPUFORT_ARRAY_PRINT_PREC[default=6]]
     * \note This method only makes sense for numeric data types and
     *       may result in compilation, runtime errors or undefined output if the
     *       underlying data is not numeric.
     */
    __host__ void print_device_data(
        std::ostream& out, const char* prefix, PrintMode print_mode,
        const int print_prec=GPUFORT_ARRAY_PRINT_PREC) const {
      const bool print_norms = 
	print_mode==PrintMode::PrintNorms ||
        print_mode==PrintMode::PrintValuesAndNorms ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      const bool print_values = 
        print_mode==PrintMode::PrintValues ||
        print_mode==PrintMode::PrintValuesAndNorms;
      const bool print_values_c_style = 
        print_mode==PrintMode::PrintCValues ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      if ( !print_norms && !print_values && !print_values_c_style ) {
        std::cerr << "ERROR: gpufort::array3::print_device_data(...): Unexpected value for 'print_mode': " 
                  << static_cast<int>(print_mode) << std::endl; 
        std::terminate();
      }
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1); 
      const int n2  = this->size(2);
      const int lb2 = this->lbound(2); 
      const int n3  = this->size(3);
      const int lb3 = this->lbound(3); 
      const int n = this->data.num_elements; 
      std::vector<T> host_array(n);
      T* A_h = host_array.data();
      T* A   = this->data.data_dev;
      HIP_CHECK(hipMemcpy(A_h, A, n*sizeof(T), hipMemcpyDeviceToHost));
      T min = +std::numeric_limits<T>::max();
      T max = -std::numeric_limits<T>::max();
      T sum = 0;
      T l1  = 0;
      T l2  = 0;
      out << prefix << ":\n";
    for ( int i3 = 0; i3 < n3; i3++ ) {
    for ( int i2 = 0; i2 < n2; i2++ ) {
    for ( int i1 = 0; i1 < n1; i1++ ) {
      const int idx = this->linearized_index(i1,i2,i3);
      T value = A_h[idx];
      if ( print_norms ) {
        min  = std::min(value,min);
        max  = std::max(value,max);
        sum += value;
        l1  += std::abs(value);
        l2  += value*value;
      }
      if ( print_values ) {
          out << prefix << "(" << (lb1+i1) << "," << (lb2+i2) << "," << (lb3+i3) <<  ") = " << std::setprecision(print_prec) << value << "\n";
      } else if ( print_values_c_style ) {
          out << prefix << "[" << idx << "] = " << std::setprecision(print_prec) << value << "\n";
      }
    }}} // for loops
    if ( print_norms ) {
      out << prefix << ":min=" << std::setprecision(print_prec) << min << "\n";
      out << prefix << ":max=" << std::setprecision(print_prec) << max << "\n";
      out << prefix << ":sum=" << std::setprecision(print_prec) << sum << "\n";
      out << prefix << ":l1="  << std::setprecision(print_prec) << l1  << "\n";
      out << prefix << ":l2="  << std::setprecision(print_prec) << std::sqrt(l2) << "\n";
      out << prefix << "num_elements=" << std::setprecision(print_prec) << n << "\n";
    }
  }

  };

  /**
   * Intended to be passed as kernel launch parameter.
   * \note Operator `T& operator()` requires parametrization with type T.
   * \note Size of this struct independent of template type T.
   */
  template<typename T>
  struct array_descr4 {
    T*     data_host = nullptr;
    T*     data_dev  = nullptr;
    int num_elements = 0;     //> Number of elements represented by this array.
    int index_offset = -1;    //> Offset for index calculation; scalar product of negative lower bounds and strides.
    int    stride2  = -1;    //> Stride 2 for linearizing 4-dimensional index.
    int    stride3  = -1;    //> Stride 3 for linearizing 4-dimensional index.
    int    stride4  = -1;    //> Stride 4 for linearizing 4-dimensional index.
 
    /**
     * Wrap a host-device pointer pair and store the associated
     * array meta data.
     *
     * \param[in] data_host host data pointer.
     * \param[in] data_dev device data pointer.
     * \param[in] n1,n2,... element counts per dimension of the multi-dim. array 
     * \param[in] lb1,lb2,... lower bounds, i.e. the index of the first element per dimension of the multi-dim. array 
     */ 
    __host__ __device__ void wrap(
        T* data_host,
        T* data_dev,
        const int n1,
        const int n2,
        const int n3,
        const int n4,
        const int lb1,
        const int lb2,
        const int lb3,
        const int lb4
      ) {
       this->data_host = data_host;
       this->data_dev  = data_dev;
       // column-major access
       this->stride2 = 1*n1;
       this->stride3 = this->stride2*n2;
       this->stride4 = this->stride3*n3;
       this->num_elements = this->stride4*n4;
       this->index_offset =
         -lb1*1
         -lb2*this->stride2
         -lb3*this->stride3
         -lb4*this->stride4;
    }
    
    /**
     * Linearize multi-dimensional index.
     *
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ int linearized_index (
      const int i1,
      const int i2,
      const int i3,
      const int i4
    ) const {
      return this->index_offset
          + i1*1
          + i2*this->stride2
          + i3*this->stride3
          + i4*this->stride4;
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ T& operator() (
      const int i1,
      const int i2,
      const int i3,
      const int i4
    ) {
      const int index = linearized_index(
        i1,i2,i3,i4
      );
      #ifdef __HIP_DEVICE_COMPILE__
      return this->data_dev[index];
      #else
      assert(i1 >= lbound(1));
      assert(i1 <= ubound(1));
      assert(i2 >= lbound(2));
      assert(i2 <= ubound(2));
      assert(i3 >= lbound(3));
      assert(i3 <= ubound(3));
      assert(i4 >= lbound(4));
      assert(i4 <= ubound(4));
      return this->data_host[index];
      #endif
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ const T& operator() (
      const int i1,
      const int i2,
      const int i3,
      const int i4
    ) const {
      const int index = linearized_index(
        i1,i2,i3,i4
      );
      #ifdef __HIP_DEVICE_COMPILE__
      return this->data_dev[index];
      #else
      assert(i1 >= lbound(1));
      assert(i1 <= ubound(1));
      assert(i2 >= lbound(2));
      assert(i2 <= ubound(2));
      assert(i3 >= lbound(3));
      assert(i3 <= ubound(3));
      assert(i4 >= lbound(4));
      assert(i4 <= ubound(4));
      return this->data_host[index];
      #endif
    }
    
    /**
     * \return Size of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,4
     */
    __host__ __device__ __forceinline__ int size(int dim) const {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(dim >= 1);
      assert(dim <= 4);
      #endif
      switch(dim) {
       case 4:
	        return this->num_elements / this->stride4;
       case 3:
	        return this->stride4 / this->stride3;
       case 2:
	        return this->stride3 / this->stride2;
       case 1:
	        return this->stride2 / 1;
       default:
          #ifndef __HIP_DEVICE_COMPILE__
          std::cerr << "‘dim’ argument of ‘gpufort::array4::size’ is not a valid dimension index ('dim': "<<dim<<", max dimension: 4" << std::endl;
          std::terminate();
          #else
          return -1;
          #endif
      }
    }
    
    /**
     * \return Lower bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,4
     */
    __host__ __device__ __forceinline__ int lbound(int dim) const {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(dim >= 1);
      assert(dim <= 4);
      #endif
      int offset_remainder = this->index_offset;
      int lb = -offset_remainder / this->stride4;
      if ( dim == 4 ) return lb;
      offset_remainder = offset_remainder + lb*this->stride4;
      lb = -offset_remainder / this->stride3;
      if ( dim == 3 ) return lb;
      offset_remainder = offset_remainder + lb*this->stride3;
      lb = -offset_remainder / this->stride2;
      if ( dim == 2 ) return lb;
      offset_remainder = offset_remainder + lb*this->stride2;
      lb = -offset_remainder / 1;
      #ifndef __HIP_DEVICE_COMPILE__
      offset_remainder = offset_remainder + lb*1;
      assert(offset_remainder == 0);
      #endif
      return lb;
    }
    
    /**
     * \return Upper bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,4
     */
    __host__ __device__ __forceinline__ int ubound(int dim) const {
      return this->lbound(dim) + this->size(dim) - 1;
    }
  };

  template<typename T>
  struct array4{
    array_descr4<T> data;
    AllocMode              alloc_mode = AllocMode::WrapHostAllocDevice; //> Data allocation strategy. Default: 
                                                                        //> wrap the host and allocate device data
    SyncMode               sync_mode  = SyncMode::None;                 //> How data should be synchronized
                                                                        //> during the initialization and destruction of this GPUFORT array.
    int num_refs             = 0;  //> Number of references.
    size_t bytes_per_element = -1; //> Bytes per element; stored to make num_data_bytes routine independent of T 

    array4() {
      // do nothing
    }
    
    ~array4() {
      // do nothing
    }

    /**
     * Initialize.
     * \param[in] data_host host data pointer (may be nullptr; see the note).
     * \param[in] pinned the host array is pinned or should be pinned.
     * \param[in] copyout_at_destruction copy data out from the
     *            device to the host when this instance gets destroyed.
     * \note Allocates a new host_data pointer if the
     * passed pointer is a nullptr. If the pinned argument is true,
     * the new hostpotr is allocated via hipHostMalloc.
     * Otherwise, it is allocated via classic malloc.
     */
    __host__ hipError_t init_async(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int n1,int n2,int n3,int n4,
        int lb1,int lb2,int lb3,int lb4,
        hipStream_t stream,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      this->data.wrap(
          data_host,
          data_dev,
          n1,n2,n3,n4,
          lb1,lb2,lb3,lb4
      );
      this->bytes_per_element = bytes_per_element;
      this->num_refs          = 1;
      this->alloc_mode        = alloc_mode;
      this->sync_mode         = sync_mode;
  
      // host array allocation
      hipError_t ierr = hipSuccess;
      switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
#endif
        case AllocMode::AllocPinnedHostWrapDevice:
        case AllocMode::AllocPinnedHostAllocDevice:
          ierr = hipHostMalloc((void**) &this->data.data_host,this->num_data_bytes(),0);
	  break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
          this->data.data_host = (T*) malloc(this->num_data_bytes());
	  break;
#endif
	case AllocMode::WrapHostWrapDevice:
        case AllocMode::WrapHostAllocDevice:
          // do nothing as already set by data.wrap call
	  break;
	default:
          std::cerr << "ERROR: gpufort::array4::init_async(...): Unexpected value for 'this->alloc_mode': " 
                    << static_cast<int>(this->alloc_mode) << std::endl; 
          std::terminate();
          break;
      }
      // device array allocation
      if ( ierr == hipSuccess ) {
        switch (this->alloc_mode) {
          case AllocMode::AllocHostWrapDevice:
          case AllocMode::AllocPinnedHostWrapDevice:
          case AllocMode::WrapHostWrapDevice:
            // do nothing as already set by data.wrap call
            break;
          case AllocMode::WrapHostAllocDevice:
          case AllocMode::AllocHostAllocDevice:
          case AllocMode::AllocPinnedHostAllocDevice:
            ierr = hipMalloc((void**) &this->data.data_dev, this->num_data_bytes());
            break;
          default:
            break;
        }
      }
      if ( ierr == hipSuccess ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::None:
          case SyncMode::Copyout:
          case SyncMode::InvertedCopyout:
            // do nothing
            break;
          case SyncMode::Copy:
          case SyncMode::Copyin:
            ierr = this->copy_to_device_async(stream);
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyin:
            ierr = this->copy_to_host_async(stream);
            break;
          default:
            std::cerr << "ERROR: gpufort::array4::init_async(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
      }
      return ierr;
    }
   
    __host__ hipError_t init_async(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int* sizes,
        int* lower_bounds,
        hipStream_t stream,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      return this->init_async(
        bytes_per_element,
        data_host,
        data_dev,
        sizes[0],sizes[1],sizes[2],sizes[3],
        lower_bounds[0],lower_bounds[1],lower_bounds[2],lower_bounds[3],
        stream,alloc_mode,sync_mode);
    }
    
    /**
     * Destroy associated host and device data
     * if this is not a wrapper. Depending
     * on the pinned attribue, hipHostFree 
     * is used instead of free to deallocate
     * the host data.
     */
    __host__ hipError_t destroy_async(hipStream_t stream) {
      hipError_t ierr = hipSuccess;
      if ( bytes_per_element > 0 ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::Copy:
          case SyncMode::Copyout:
            ierr = this->copy_to_host_async(stream);
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyout:
            ierr = this->copy_to_device_async(stream);
            break;
          case SyncMode::None:
          case SyncMode::Copyin:
          case SyncMode::InvertedCopyin:
            // do nothing
            break;
          default:
            std::cerr << "ERROR: gpufort::array4::destroy_async(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
        // host array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
#endif
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipHostFree(this->data.data_host);
              break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
              free(this->data.data_host); 
              break;
#endif
            case AllocMode::WrapHostWrapDevice:
            case AllocMode::WrapHostAllocDevice:
              break;
            default:
              std::cerr << "ERROR: gpufort::array4::destroy_async(...): Unexpected value for 'alloc_mode': " 
                        << static_cast<int>(alloc_mode) << std::endl; 
              std::terminate();
              break;
          }
        }
        // device array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::WrapHostWrapDevice:
              // do nothing as already set by data.wrap call
              break;
            case AllocMode::WrapHostAllocDevice:
            case AllocMode::AllocHostAllocDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipFree(this->data.data_dev);
              break;
            default:
              break;
          }
        }
        if ( ierr == hipSuccess ) {
          this->data.data_host    = nullptr;
          this->data.data_dev     = nullptr;
          this->bytes_per_element = -1; 
        } 
      }
      return ierr;
    }
    
    /**
     * Copy data to a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the destinaton buffer is interpreted as host or device variable
     * and the source is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_to_buffer_async(
       void* buffer, hipMemcpyKind memcpy_kind,
       hipStream_t stream
    ) {
      T* src = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyDeviceToHost: 
        case hipMemcpyDeviceToDevice:
          src = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyHostToDevice:
          src = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array4::copy_to_buffer_async(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(src!=nullptr);
      #endif
      return hipMemcpyAsync(
        buffer, 
        (void*) src,
        this->num_data_bytes(), 
        memcpy_kind, stream);
    }
    
    /**
     * Copy data from a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the source buffer is interpreted as host or device variable
     * and the destination is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_from_buffer_async(
       void* buffer, hipMemcpyKind memcpy_kind,
       hipStream_t stream
    ) {
      T* dest = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyHostToDevice: 
        case hipMemcpyDeviceToDevice:
          dest = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyDeviceToHost:
          dest = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array4::copy_from_buffer_async(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(dest!=nullptr);
      #endif
      return hipMemcpyAsync(
        (void*) dest,
        buffer, 
        this->num_data_bytes(), 
        memcpy_kind, stream);
    }
    
    /**
     * Copy host data to the device.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_host_async(hipStream_t stream) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpyAsync(
        (void*) this->data.data_host, 
        (void*) this->data.data_dev,
        this->num_data_bytes(), 
        hipMemcpyDeviceToHost, stream);
    }
    
    /**
     * Copy device data to the host.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_device_async(hipStream_t stream) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpyAsync(
        (void*) this->data.data_dev, 
        (void*) this->data.data_host,
        this->num_data_bytes(),
        hipMemcpyHostToDevice, stream);
    }
    
    /** 
     * Copies the struct to the device into an pre-allocated
     * device memory block but does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[in] device_struct device memory address to copy to
     */
    __host__ hipError_t copy_self_to_device_async(
        gpufort::array4<T>* device_struct,
        hipStream_t stream
    ) {
      const size_t size = sizeof(array4<char>); // sizeof(T*) = sizeof(char*)
      return hipMemcpyAsync(
          (void*) device_struct, 
          (void*) this,
          size,
          hipMemcpyHostToDevice, stream);
    }

    /** 
     * Copies the struct to the device but
     * does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[inout] device_copy pointer to device copy pointer 
     */
    __host__ hipError_t create_device_copy_async(
        gpufort::array4<T>** device_copy,
        hipStream_t stream
    ) {
      const size_t size = sizeof(array4<char>); // sizeof(T*) = sizeof(char*)
      hipError_t ierr   = hipMalloc((void**)device_copy,size);      
      if ( ierr == hipSuccess ) {
        return this->copy_self_to_device_async(*device_copy,stream);
      } else {
        return ierr;
      }
    }
    /**
     * Initialize.
     * \param[in] data_host host data pointer (may be nullptr; see the note).
     * \param[in] pinned the host array is pinned or should be pinned.
     * \param[in] copyout_at_destruction copy data out from the
     *            device to the host when this instance gets destroyed.
     * \note Allocates a new host_data pointer if the
     * passed pointer is a nullptr. If the pinned argument is true,
     * the new hostpotr is allocated via hipHostMalloc.
     * Otherwise, it is allocated via classic malloc.
     */
    __host__ hipError_t init(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int n1,int n2,int n3,int n4,
        int lb1,int lb2,int lb3,int lb4,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      this->data.wrap(
          data_host,
          data_dev,
          n1,n2,n3,n4,
          lb1,lb2,lb3,lb4
      );
      this->bytes_per_element = bytes_per_element;
      this->num_refs          = 1;
      this->alloc_mode        = alloc_mode;
      this->sync_mode         = sync_mode;
  
      // host array allocation
      hipError_t ierr = hipSuccess;
      switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
#endif
        case AllocMode::AllocPinnedHostWrapDevice:
        case AllocMode::AllocPinnedHostAllocDevice:
          ierr = hipHostMalloc((void**) &this->data.data_host,this->num_data_bytes(),0);
	  break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
          this->data.data_host = (T*) malloc(this->num_data_bytes());
	  break;
#endif
	case AllocMode::WrapHostWrapDevice:
        case AllocMode::WrapHostAllocDevice:
          // do nothing as already set by data.wrap call
	  break;
	default:
          std::cerr << "ERROR: gpufort::array4::init(...): Unexpected value for 'this->alloc_mode': " 
                    << static_cast<int>(this->alloc_mode) << std::endl; 
          std::terminate();
          break;
      }
      // device array allocation
      if ( ierr == hipSuccess ) {
        switch (this->alloc_mode) {
          case AllocMode::AllocHostWrapDevice:
          case AllocMode::AllocPinnedHostWrapDevice:
          case AllocMode::WrapHostWrapDevice:
            // do nothing as already set by data.wrap call
            break;
          case AllocMode::WrapHostAllocDevice:
          case AllocMode::AllocHostAllocDevice:
          case AllocMode::AllocPinnedHostAllocDevice:
            ierr = hipMalloc((void**) &this->data.data_dev, this->num_data_bytes());
            break;
          default:
            break;
        }
      }
      if ( ierr == hipSuccess ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::None:
          case SyncMode::Copyout:
          case SyncMode::InvertedCopyout:
            // do nothing
            break;
          case SyncMode::Copy:
          case SyncMode::Copyin:
            ierr = this->copy_to_device();
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyin:
            ierr = this->copy_to_host();
            break;
          default:
            std::cerr << "ERROR: gpufort::array4::init(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
      }
      return ierr;
    }
   
    __host__ hipError_t init(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int* sizes,
        int* lower_bounds,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      return this->init(
        bytes_per_element,
        data_host,
        data_dev,
        sizes[0],sizes[1],sizes[2],sizes[3],
        lower_bounds[0],lower_bounds[1],lower_bounds[2],lower_bounds[3],alloc_mode,sync_mode);
    }
    
    /**
     * Destroy associated host and device data
     * if this is not a wrapper. Depending
     * on the pinned attribue, hipHostFree 
     * is used instead of free to deallocate
     * the host data.
     */
    __host__ hipError_t destroy() {
      hipError_t ierr = hipSuccess;
      if ( bytes_per_element > 0 ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::Copy:
          case SyncMode::Copyout:
            ierr = this->copy_to_host();
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyout:
            ierr = this->copy_to_device();
            break;
          case SyncMode::None:
          case SyncMode::Copyin:
          case SyncMode::InvertedCopyin:
            // do nothing
            break;
          default:
            std::cerr << "ERROR: gpufort::array4::destroy(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
        // host array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
#endif
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipHostFree(this->data.data_host);
              break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
              free(this->data.data_host); 
              break;
#endif
            case AllocMode::WrapHostWrapDevice:
            case AllocMode::WrapHostAllocDevice:
              break;
            default:
              std::cerr << "ERROR: gpufort::array4::destroy(...): Unexpected value for 'alloc_mode': " 
                        << static_cast<int>(alloc_mode) << std::endl; 
              std::terminate();
              break;
          }
        }
        // device array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::WrapHostWrapDevice:
              // do nothing as already set by data.wrap call
              break;
            case AllocMode::WrapHostAllocDevice:
            case AllocMode::AllocHostAllocDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipFree(this->data.data_dev);
              break;
            default:
              break;
          }
        }
        if ( ierr == hipSuccess ) {
          this->data.data_host    = nullptr;
          this->data.data_dev     = nullptr;
          this->bytes_per_element = -1; 
        } 
      }
      return ierr;
    }
    
    /**
     * Copy data to a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the destinaton buffer is interpreted as host or device variable
     * and the source is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_to_buffer(
       void* buffer, hipMemcpyKind memcpy_kind
    ) {
      T* src = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyDeviceToHost: 
        case hipMemcpyDeviceToDevice:
          src = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyHostToDevice:
          src = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array4::copy_to_buffer(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(src!=nullptr);
      #endif
      return hipMemcpy(
        buffer, 
        (void*) src,
        this->num_data_bytes(), 
        memcpy_kind);
    }
    
    /**
     * Copy data from a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the source buffer is interpreted as host or device variable
     * and the destination is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_from_buffer(
       void* buffer, hipMemcpyKind memcpy_kind
    ) {
      T* dest = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyHostToDevice: 
        case hipMemcpyDeviceToDevice:
          dest = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyDeviceToHost:
          dest = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array4::copy_from_buffer(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(dest!=nullptr);
      #endif
      return hipMemcpy(
        (void*) dest,
        buffer, 
        this->num_data_bytes(), 
        memcpy_kind);
    }
    
    /**
     * Copy host data to the device.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_host() {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpy(
        (void*) this->data.data_host, 
        (void*) this->data.data_dev,
        this->num_data_bytes(), 
        hipMemcpyDeviceToHost);
    }
    
    /**
     * Copy device data to the host.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_device() {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpy(
        (void*) this->data.data_dev, 
        (void*) this->data.data_host,
        this->num_data_bytes(),
        hipMemcpyHostToDevice);
    }
    
    /** 
     * Copies the struct to the device into an pre-allocated
     * device memory block but does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[in] device_struct device memory address to copy to
     */
    __host__ hipError_t copy_self_to_device(
        gpufort::array4<T>* device_struct
    ) {
      const size_t size = sizeof(array4<char>); // sizeof(T*) = sizeof(char*)
      return hipMemcpy(
          (void*) device_struct, 
          (void*) this,
          size,
          hipMemcpyHostToDevice);
    }

    /** 
     * Copies the struct to the device but
     * does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[inout] device_copy pointer to device copy pointer 
     */
    __host__ hipError_t create_device_copy(
        gpufort::array4<T>** device_copy
    ) {
      const size_t size = sizeof(array4<char>); // sizeof(T*) = sizeof(char*)
      hipError_t ierr   = hipMalloc((void**)device_copy,size);      
      if ( ierr == hipSuccess ) {
        return this->copy_self_to_device(*device_copy);
      } else {
        return ierr;
      }
    }

    /**
     * Allocate a host buffer with
     * the same size as the data buffers
     * associated with this gpufort array.
     * @see num_data_bytes()
     * \param[inout] pointer to the buffer to allocate
     * \param[in] pinned If the memory should be pinned (default=True)
     * \param[in] flags  Flags for the host memory allocation (default=0).
     */
    __host__ hipError_t allocate_host_buffer(void** buffer,bool pinned=true,int flags=0) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
      if ( pinned ) { 
        ierr = hipHostMalloc(buffer,this->num_data_bytes(),flags);
      } else {
        *buffer = malloc(this->num_data_bytes());
      }
      return ierr;
    }

    /**
     * Deallocate a host buffer
     * created via the allocate_host_buffer routine.
     * @see num_data_bytes(), allocate_host_buffer
     * \param[inout] the buffer to deallocte
     * \param[in] pinned If the memory to deallocate is pinned (default=True)
     */
    __host__ hipError_t deallocate_host_buffer(void* buffer,bool pinned=true) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
      if ( pinned ) { 
        ierr = hipHostFree(buffer);
      } else {
        free(buffer);
      }
      return ierr;
    }
    /**
     * Allocate a device buffer with
     * the same size as the data buffers
     * associated with this gpufort array.
     * @see num_data_bytes()
     * \param[inout] pointer to the buffer to allocate
     */
    __host__ hipError_t allocate_device_buffer(void** buffer) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
 
      ierr = hipMalloc(buffer,this->num_data_bytes());
      return ierr;
    }

    /**
     * Deallocate a device buffer
     * created via the allocate_device_buffer routine.
     * @see num_data_bytes(), allocate_device_buffer
     * \param[inout] the buffer to deallocte
     */
    __host__ hipError_t deallocate_device_buffer(void* buffer) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
 
      ierr = hipFree(buffer);
      return ierr;
    }

    /** 
     * Copy metadata from another gpufort array to this gpufort array.
     * \note The allocation mode is always set to AllocMode::WrapHostWrapDevice
     *       and the sync mode is set to SyncMode::None.
     * \note No deep copy, i.e. the host and device data is not copied, just the pointers
     * to the data.
     */
    __host__ void copy(const gpufort::array4<T>& other) {
      this->init(
        other.bytes_per_element,
        other.data.data_host,
        other.data.data_dev,
        other.size(1),other.size(2),other.size(3),other.size(4),
        other.lbound(1),other.lbound(2),other.lbound(3),other.lbound(4),
        AllocMode::WrapHostWrapDevice,
        SyncMode::None);
    }

    /** 
     *  \return Number of bytes required to store 
     *  the array elements on the host or the device.
     */ 
    __host__ size_t num_data_bytes() {
      return this->data.num_elements * this->bytes_per_element;
    }
    
    /**
     * \return the number of elements of this array.
     */
    __host__ int num_elements() {
      return this->data.num_elements;
    }

    /**
     * Linearize multi-dimensional index.
     *
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ int linearized_index (
      const int i1,
      const int i2,
      const int i3,
      const int i4
    ) const {
      return this->data.linearized_index(
        i1,i2,i3,i4
      );
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ T& operator() (
      const int i1,
      const int i2,
      const int i3,
      const int i4
    ) {
      return this->data(
        i1,i2,i3,i4
      );
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ const T& operator() (
      const int i1,
      const int i2,
      const int i3,
      const int i4
    ) const {
      return this->data(
        i1,i2,i3,i4
      );
    }
    
    /**
     * \return Size of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,4
     */
    __host__ __device__ __forceinline__ int size(int dim) const {
      return this->data.size(dim);
    }
    
    /**
     * \return Lower bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,4
     */
    __host__ __device__ __forceinline__ int lbound(int dim) const {
      return this->data.lbound(dim);
    }
    
    /**
     * \return Upper bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,4
     */
    __host__ __device__ __forceinline__ int ubound(int dim) const {
      return this->data.ubound(dim);
    }
    

    /**
     * Collapse the array by fixing 1 indices.
     * \return A gpufort array of rank 3.
     * \param[in] i4,...,i4 indices to fix.
     */
    __host__ gpufort::array3<T> collapse(
        const int i4
    ) const {
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1);
      const int n2  = this->size(2);
      const int lb2 = this->lbound(2);
      const int n3  = this->size(3);
      const int lb3 = this->lbound(3);
      const int index = linearized_index(
        lb1,lb2,lb3,
        i4);
      T* data_host_new = nullptr; 
      if (this->data.data_host != nullptr) {
        data_host_new = this->data.data_host + index;
      }
      T* data_dev_new = nullptr; 
      if (this->data.data_dev != nullptr) {
        data_dev_new = this->data.data_dev + index;
      }
      gpufort::array3<T> collapsed_array;
      hipError_t ierr = 
        collapsed_array.init(
          this->bytes_per_element,
          data_host_new,
          data_dev_new,
          n1,n2,n3,
          lb1,lb2,lb3,
          AllocMode::WrapHostWrapDevice,
          SyncMode::None);
      return collapsed_array;
    }
    /**
     * Collapse the array by fixing 2 indices.
     * \return A gpufort array of rank 2.
     * \param[in] i3,...,i4 indices to fix.
     */
    __host__ gpufort::array2<T> collapse(
        const int i3,
        const int i4
    ) const {
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1);
      const int n2  = this->size(2);
      const int lb2 = this->lbound(2);
      const int index = linearized_index(
        lb1,lb2,
        i3,i4);
      T* data_host_new = nullptr; 
      if (this->data.data_host != nullptr) {
        data_host_new = this->data.data_host + index;
      }
      T* data_dev_new = nullptr; 
      if (this->data.data_dev != nullptr) {
        data_dev_new = this->data.data_dev + index;
      }
      gpufort::array2<T> collapsed_array;
      hipError_t ierr = 
        collapsed_array.init(
          this->bytes_per_element,
          data_host_new,
          data_dev_new,
          n1,n2,
          lb1,lb2,
          AllocMode::WrapHostWrapDevice,
          SyncMode::None);
      return collapsed_array;
    }
    /**
     * Collapse the array by fixing 3 indices.
     * \return A gpufort array of rank 1.
     * \param[in] i2,...,i4 indices to fix.
     */
    __host__ gpufort::array1<T> collapse(
        const int i2,
        const int i3,
        const int i4
    ) const {
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1);
      const int index = linearized_index(
        lb1,
        i2,i3,i4);
      T* data_host_new = nullptr; 
      if (this->data.data_host != nullptr) {
        data_host_new = this->data.data_host + index;
      }
      T* data_dev_new = nullptr; 
      if (this->data.data_dev != nullptr) {
        data_dev_new = this->data.data_dev + index;
      }
      gpufort::array1<T> collapsed_array;
      hipError_t ierr = 
        collapsed_array.init(
          this->bytes_per_element,
          data_host_new,
          data_dev_new,
          n1,
          lb1,
          AllocMode::WrapHostWrapDevice,
          SyncMode::None);
      return collapsed_array;
    }

    /**
     * Write host data values and their corresponding index
     * to an output stream.
     * \param[inout] out        output stream
     * \param[in]    prefix     prefix to put before each output line
     * \param[in]    print_prec precision to use when printing floating point numbers 
     *               [default=GPUFORT_ARRAY_PRINT_PREC[default=6]]
     * \note This method only makes sense for numeric data types and
     *       may result in compilation, runtime errors or undefined output if the
     *       underlying data is not numeric.
     */
    __host__ void print_host_data(
        std::ostream& out, const char* prefix, PrintMode print_mode,
        const int print_prec=GPUFORT_ARRAY_PRINT_PREC) const {
      const bool print_norms = 
	print_mode==PrintMode::PrintNorms ||
        print_mode==PrintMode::PrintValuesAndNorms ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      const bool print_values = 
        print_mode==PrintMode::PrintValues ||
        print_mode==PrintMode::PrintValuesAndNorms;
      const bool print_values_c_style = 
        print_mode==PrintMode::PrintCValues ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      if ( !print_norms && !print_values && !print_values_c_style ) {
        std::cerr << "ERROR: gpufort::array4::print_host_data(...): Unexpected value for 'print_mode': " 
                  << static_cast<int>(print_mode) << std::endl; 
        std::terminate();
      }
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1); 
      const int n2  = this->size(2);
      const int lb2 = this->lbound(2); 
      const int n3  = this->size(3);
      const int lb3 = this->lbound(3); 
      const int n4  = this->size(4);
      const int lb4 = this->lbound(4); 
      const int n = this->data.num_elements; 
      T* A_h   = this->data.data_host;
      T min = +std::numeric_limits<T>::max();
      T max = -std::numeric_limits<T>::max();
      T sum = 0;
      T l1  = 0;
      T l2  = 0;
      out << prefix << ":\n";
    for ( int i4 = 0; i4 < n4; i4++ ) {
    for ( int i3 = 0; i3 < n3; i3++ ) {
    for ( int i2 = 0; i2 < n2; i2++ ) {
    for ( int i1 = 0; i1 < n1; i1++ ) {
      const int idx = this->linearized_index(i1,i2,i3,i4);
      T value = A_h[idx];
      if ( print_norms ) {
        min  = std::min(value,min);
        max  = std::max(value,max);
        sum += value;
        l1  += std::abs(value);
        l2  += value*value;
      }
      if ( print_values ) {
          out << prefix << "(" << (lb1+i1) << "," << (lb2+i2) << "," << (lb3+i3) << "," << (lb4+i4) <<  ") = " << std::setprecision(print_prec) << value << "\n";
      } else if ( print_values_c_style ) {
          out << prefix << "[" << idx << "] = " << std::setprecision(print_prec) << value << "\n";
      }
    }}}} // for loops
    if ( print_norms ) {
      out << prefix << ":min=" << std::setprecision(print_prec) << min << "\n";
      out << prefix << ":max=" << std::setprecision(print_prec) << max << "\n";
      out << prefix << ":sum=" << std::setprecision(print_prec) << sum << "\n";
      out << prefix << ":l1="  << std::setprecision(print_prec) << l1  << "\n";
      out << prefix << ":l2="  << std::setprecision(print_prec) << std::sqrt(l2) << "\n";
      out << prefix << "num_elements=" << std::setprecision(print_prec) << n << "\n";
    }
  }
    /**
     * Write device data values and their corresponding index
     * to an output stream.
     * \param[inout] out        output stream
     * \param[in]    prefix     prefix to put before each output line
     * \param[in]    print_prec precision to use when printing floating point numbers 
     *               [default=GPUFORT_ARRAY_PRINT_PREC[default=6]]
     * \note This method only makes sense for numeric data types and
     *       may result in compilation, runtime errors or undefined output if the
     *       underlying data is not numeric.
     */
    __host__ void print_device_data(
        std::ostream& out, const char* prefix, PrintMode print_mode,
        const int print_prec=GPUFORT_ARRAY_PRINT_PREC) const {
      const bool print_norms = 
	print_mode==PrintMode::PrintNorms ||
        print_mode==PrintMode::PrintValuesAndNorms ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      const bool print_values = 
        print_mode==PrintMode::PrintValues ||
        print_mode==PrintMode::PrintValuesAndNorms;
      const bool print_values_c_style = 
        print_mode==PrintMode::PrintCValues ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      if ( !print_norms && !print_values && !print_values_c_style ) {
        std::cerr << "ERROR: gpufort::array4::print_device_data(...): Unexpected value for 'print_mode': " 
                  << static_cast<int>(print_mode) << std::endl; 
        std::terminate();
      }
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1); 
      const int n2  = this->size(2);
      const int lb2 = this->lbound(2); 
      const int n3  = this->size(3);
      const int lb3 = this->lbound(3); 
      const int n4  = this->size(4);
      const int lb4 = this->lbound(4); 
      const int n = this->data.num_elements; 
      std::vector<T> host_array(n);
      T* A_h = host_array.data();
      T* A   = this->data.data_dev;
      HIP_CHECK(hipMemcpy(A_h, A, n*sizeof(T), hipMemcpyDeviceToHost));
      T min = +std::numeric_limits<T>::max();
      T max = -std::numeric_limits<T>::max();
      T sum = 0;
      T l1  = 0;
      T l2  = 0;
      out << prefix << ":\n";
    for ( int i4 = 0; i4 < n4; i4++ ) {
    for ( int i3 = 0; i3 < n3; i3++ ) {
    for ( int i2 = 0; i2 < n2; i2++ ) {
    for ( int i1 = 0; i1 < n1; i1++ ) {
      const int idx = this->linearized_index(i1,i2,i3,i4);
      T value = A_h[idx];
      if ( print_norms ) {
        min  = std::min(value,min);
        max  = std::max(value,max);
        sum += value;
        l1  += std::abs(value);
        l2  += value*value;
      }
      if ( print_values ) {
          out << prefix << "(" << (lb1+i1) << "," << (lb2+i2) << "," << (lb3+i3) << "," << (lb4+i4) <<  ") = " << std::setprecision(print_prec) << value << "\n";
      } else if ( print_values_c_style ) {
          out << prefix << "[" << idx << "] = " << std::setprecision(print_prec) << value << "\n";
      }
    }}}} // for loops
    if ( print_norms ) {
      out << prefix << ":min=" << std::setprecision(print_prec) << min << "\n";
      out << prefix << ":max=" << std::setprecision(print_prec) << max << "\n";
      out << prefix << ":sum=" << std::setprecision(print_prec) << sum << "\n";
      out << prefix << ":l1="  << std::setprecision(print_prec) << l1  << "\n";
      out << prefix << ":l2="  << std::setprecision(print_prec) << std::sqrt(l2) << "\n";
      out << prefix << "num_elements=" << std::setprecision(print_prec) << n << "\n";
    }
  }

  };

  /**
   * Intended to be passed as kernel launch parameter.
   * \note Operator `T& operator()` requires parametrization with type T.
   * \note Size of this struct independent of template type T.
   */
  template<typename T>
  struct array_descr5 {
    T*     data_host = nullptr;
    T*     data_dev  = nullptr;
    int num_elements = 0;     //> Number of elements represented by this array.
    int index_offset = -1;    //> Offset for index calculation; scalar product of negative lower bounds and strides.
    int    stride2  = -1;    //> Stride 2 for linearizing 5-dimensional index.
    int    stride3  = -1;    //> Stride 3 for linearizing 5-dimensional index.
    int    stride4  = -1;    //> Stride 4 for linearizing 5-dimensional index.
    int    stride5  = -1;    //> Stride 5 for linearizing 5-dimensional index.
 
    /**
     * Wrap a host-device pointer pair and store the associated
     * array meta data.
     *
     * \param[in] data_host host data pointer.
     * \param[in] data_dev device data pointer.
     * \param[in] n1,n2,... element counts per dimension of the multi-dim. array 
     * \param[in] lb1,lb2,... lower bounds, i.e. the index of the first element per dimension of the multi-dim. array 
     */ 
    __host__ __device__ void wrap(
        T* data_host,
        T* data_dev,
        const int n1,
        const int n2,
        const int n3,
        const int n4,
        const int n5,
        const int lb1,
        const int lb2,
        const int lb3,
        const int lb4,
        const int lb5
      ) {
       this->data_host = data_host;
       this->data_dev  = data_dev;
       // column-major access
       this->stride2 = 1*n1;
       this->stride3 = this->stride2*n2;
       this->stride4 = this->stride3*n3;
       this->stride5 = this->stride4*n4;
       this->num_elements = this->stride5*n5;
       this->index_offset =
         -lb1*1
         -lb2*this->stride2
         -lb3*this->stride3
         -lb4*this->stride4
         -lb5*this->stride5;
    }
    
    /**
     * Linearize multi-dimensional index.
     *
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ int linearized_index (
      const int i1,
      const int i2,
      const int i3,
      const int i4,
      const int i5
    ) const {
      return this->index_offset
          + i1*1
          + i2*this->stride2
          + i3*this->stride3
          + i4*this->stride4
          + i5*this->stride5;
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ T& operator() (
      const int i1,
      const int i2,
      const int i3,
      const int i4,
      const int i5
    ) {
      const int index = linearized_index(
        i1,i2,i3,i4,i5
      );
      #ifdef __HIP_DEVICE_COMPILE__
      return this->data_dev[index];
      #else
      assert(i1 >= lbound(1));
      assert(i1 <= ubound(1));
      assert(i2 >= lbound(2));
      assert(i2 <= ubound(2));
      assert(i3 >= lbound(3));
      assert(i3 <= ubound(3));
      assert(i4 >= lbound(4));
      assert(i4 <= ubound(4));
      assert(i5 >= lbound(5));
      assert(i5 <= ubound(5));
      return this->data_host[index];
      #endif
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ const T& operator() (
      const int i1,
      const int i2,
      const int i3,
      const int i4,
      const int i5
    ) const {
      const int index = linearized_index(
        i1,i2,i3,i4,i5
      );
      #ifdef __HIP_DEVICE_COMPILE__
      return this->data_dev[index];
      #else
      assert(i1 >= lbound(1));
      assert(i1 <= ubound(1));
      assert(i2 >= lbound(2));
      assert(i2 <= ubound(2));
      assert(i3 >= lbound(3));
      assert(i3 <= ubound(3));
      assert(i4 >= lbound(4));
      assert(i4 <= ubound(4));
      assert(i5 >= lbound(5));
      assert(i5 <= ubound(5));
      return this->data_host[index];
      #endif
    }
    
    /**
     * \return Size of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,5
     */
    __host__ __device__ __forceinline__ int size(int dim) const {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(dim >= 1);
      assert(dim <= 5);
      #endif
      switch(dim) {
       case 5:
	        return this->num_elements / this->stride5;
       case 4:
	        return this->stride5 / this->stride4;
       case 3:
	        return this->stride4 / this->stride3;
       case 2:
	        return this->stride3 / this->stride2;
       case 1:
	        return this->stride2 / 1;
       default:
          #ifndef __HIP_DEVICE_COMPILE__
          std::cerr << "‘dim’ argument of ‘gpufort::array5::size’ is not a valid dimension index ('dim': "<<dim<<", max dimension: 5" << std::endl;
          std::terminate();
          #else
          return -1;
          #endif
      }
    }
    
    /**
     * \return Lower bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,5
     */
    __host__ __device__ __forceinline__ int lbound(int dim) const {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(dim >= 1);
      assert(dim <= 5);
      #endif
      int offset_remainder = this->index_offset;
      int lb = -offset_remainder / this->stride5;
      if ( dim == 5 ) return lb;
      offset_remainder = offset_remainder + lb*this->stride5;
      lb = -offset_remainder / this->stride4;
      if ( dim == 4 ) return lb;
      offset_remainder = offset_remainder + lb*this->stride4;
      lb = -offset_remainder / this->stride3;
      if ( dim == 3 ) return lb;
      offset_remainder = offset_remainder + lb*this->stride3;
      lb = -offset_remainder / this->stride2;
      if ( dim == 2 ) return lb;
      offset_remainder = offset_remainder + lb*this->stride2;
      lb = -offset_remainder / 1;
      #ifndef __HIP_DEVICE_COMPILE__
      offset_remainder = offset_remainder + lb*1;
      assert(offset_remainder == 0);
      #endif
      return lb;
    }
    
    /**
     * \return Upper bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,5
     */
    __host__ __device__ __forceinline__ int ubound(int dim) const {
      return this->lbound(dim) + this->size(dim) - 1;
    }
  };

  template<typename T>
  struct array5{
    array_descr5<T> data;
    AllocMode              alloc_mode = AllocMode::WrapHostAllocDevice; //> Data allocation strategy. Default: 
                                                                        //> wrap the host and allocate device data
    SyncMode               sync_mode  = SyncMode::None;                 //> How data should be synchronized
                                                                        //> during the initialization and destruction of this GPUFORT array.
    int num_refs             = 0;  //> Number of references.
    size_t bytes_per_element = -1; //> Bytes per element; stored to make num_data_bytes routine independent of T 

    array5() {
      // do nothing
    }
    
    ~array5() {
      // do nothing
    }

    /**
     * Initialize.
     * \param[in] data_host host data pointer (may be nullptr; see the note).
     * \param[in] pinned the host array is pinned or should be pinned.
     * \param[in] copyout_at_destruction copy data out from the
     *            device to the host when this instance gets destroyed.
     * \note Allocates a new host_data pointer if the
     * passed pointer is a nullptr. If the pinned argument is true,
     * the new hostpotr is allocated via hipHostMalloc.
     * Otherwise, it is allocated via classic malloc.
     */
    __host__ hipError_t init_async(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int n1,int n2,int n3,int n4,int n5,
        int lb1,int lb2,int lb3,int lb4,int lb5,
        hipStream_t stream,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      this->data.wrap(
          data_host,
          data_dev,
          n1,n2,n3,n4,n5,
          lb1,lb2,lb3,lb4,lb5
      );
      this->bytes_per_element = bytes_per_element;
      this->num_refs          = 1;
      this->alloc_mode        = alloc_mode;
      this->sync_mode         = sync_mode;
  
      // host array allocation
      hipError_t ierr = hipSuccess;
      switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
#endif
        case AllocMode::AllocPinnedHostWrapDevice:
        case AllocMode::AllocPinnedHostAllocDevice:
          ierr = hipHostMalloc((void**) &this->data.data_host,this->num_data_bytes(),0);
	  break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
          this->data.data_host = (T*) malloc(this->num_data_bytes());
	  break;
#endif
	case AllocMode::WrapHostWrapDevice:
        case AllocMode::WrapHostAllocDevice:
          // do nothing as already set by data.wrap call
	  break;
	default:
          std::cerr << "ERROR: gpufort::array5::init_async(...): Unexpected value for 'this->alloc_mode': " 
                    << static_cast<int>(this->alloc_mode) << std::endl; 
          std::terminate();
          break;
      }
      // device array allocation
      if ( ierr == hipSuccess ) {
        switch (this->alloc_mode) {
          case AllocMode::AllocHostWrapDevice:
          case AllocMode::AllocPinnedHostWrapDevice:
          case AllocMode::WrapHostWrapDevice:
            // do nothing as already set by data.wrap call
            break;
          case AllocMode::WrapHostAllocDevice:
          case AllocMode::AllocHostAllocDevice:
          case AllocMode::AllocPinnedHostAllocDevice:
            ierr = hipMalloc((void**) &this->data.data_dev, this->num_data_bytes());
            break;
          default:
            break;
        }
      }
      if ( ierr == hipSuccess ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::None:
          case SyncMode::Copyout:
          case SyncMode::InvertedCopyout:
            // do nothing
            break;
          case SyncMode::Copy:
          case SyncMode::Copyin:
            ierr = this->copy_to_device_async(stream);
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyin:
            ierr = this->copy_to_host_async(stream);
            break;
          default:
            std::cerr << "ERROR: gpufort::array5::init_async(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
      }
      return ierr;
    }
   
    __host__ hipError_t init_async(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int* sizes,
        int* lower_bounds,
        hipStream_t stream,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      return this->init_async(
        bytes_per_element,
        data_host,
        data_dev,
        sizes[0],sizes[1],sizes[2],sizes[3],sizes[4],
        lower_bounds[0],lower_bounds[1],lower_bounds[2],lower_bounds[3],lower_bounds[4],
        stream,alloc_mode,sync_mode);
    }
    
    /**
     * Destroy associated host and device data
     * if this is not a wrapper. Depending
     * on the pinned attribue, hipHostFree 
     * is used instead of free to deallocate
     * the host data.
     */
    __host__ hipError_t destroy_async(hipStream_t stream) {
      hipError_t ierr = hipSuccess;
      if ( bytes_per_element > 0 ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::Copy:
          case SyncMode::Copyout:
            ierr = this->copy_to_host_async(stream);
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyout:
            ierr = this->copy_to_device_async(stream);
            break;
          case SyncMode::None:
          case SyncMode::Copyin:
          case SyncMode::InvertedCopyin:
            // do nothing
            break;
          default:
            std::cerr << "ERROR: gpufort::array5::destroy_async(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
        // host array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
#endif
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipHostFree(this->data.data_host);
              break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
              free(this->data.data_host); 
              break;
#endif
            case AllocMode::WrapHostWrapDevice:
            case AllocMode::WrapHostAllocDevice:
              break;
            default:
              std::cerr << "ERROR: gpufort::array5::destroy_async(...): Unexpected value for 'alloc_mode': " 
                        << static_cast<int>(alloc_mode) << std::endl; 
              std::terminate();
              break;
          }
        }
        // device array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::WrapHostWrapDevice:
              // do nothing as already set by data.wrap call
              break;
            case AllocMode::WrapHostAllocDevice:
            case AllocMode::AllocHostAllocDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipFree(this->data.data_dev);
              break;
            default:
              break;
          }
        }
        if ( ierr == hipSuccess ) {
          this->data.data_host    = nullptr;
          this->data.data_dev     = nullptr;
          this->bytes_per_element = -1; 
        } 
      }
      return ierr;
    }
    
    /**
     * Copy data to a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the destinaton buffer is interpreted as host or device variable
     * and the source is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_to_buffer_async(
       void* buffer, hipMemcpyKind memcpy_kind,
       hipStream_t stream
    ) {
      T* src = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyDeviceToHost: 
        case hipMemcpyDeviceToDevice:
          src = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyHostToDevice:
          src = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array5::copy_to_buffer_async(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(src!=nullptr);
      #endif
      return hipMemcpyAsync(
        buffer, 
        (void*) src,
        this->num_data_bytes(), 
        memcpy_kind, stream);
    }
    
    /**
     * Copy data from a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the source buffer is interpreted as host or device variable
     * and the destination is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_from_buffer_async(
       void* buffer, hipMemcpyKind memcpy_kind,
       hipStream_t stream
    ) {
      T* dest = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyHostToDevice: 
        case hipMemcpyDeviceToDevice:
          dest = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyDeviceToHost:
          dest = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array5::copy_from_buffer_async(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(dest!=nullptr);
      #endif
      return hipMemcpyAsync(
        (void*) dest,
        buffer, 
        this->num_data_bytes(), 
        memcpy_kind, stream);
    }
    
    /**
     * Copy host data to the device.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_host_async(hipStream_t stream) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpyAsync(
        (void*) this->data.data_host, 
        (void*) this->data.data_dev,
        this->num_data_bytes(), 
        hipMemcpyDeviceToHost, stream);
    }
    
    /**
     * Copy device data to the host.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_device_async(hipStream_t stream) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpyAsync(
        (void*) this->data.data_dev, 
        (void*) this->data.data_host,
        this->num_data_bytes(),
        hipMemcpyHostToDevice, stream);
    }
    
    /** 
     * Copies the struct to the device into an pre-allocated
     * device memory block but does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[in] device_struct device memory address to copy to
     */
    __host__ hipError_t copy_self_to_device_async(
        gpufort::array5<T>* device_struct,
        hipStream_t stream
    ) {
      const size_t size = sizeof(array5<char>); // sizeof(T*) = sizeof(char*)
      return hipMemcpyAsync(
          (void*) device_struct, 
          (void*) this,
          size,
          hipMemcpyHostToDevice, stream);
    }

    /** 
     * Copies the struct to the device but
     * does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[inout] device_copy pointer to device copy pointer 
     */
    __host__ hipError_t create_device_copy_async(
        gpufort::array5<T>** device_copy,
        hipStream_t stream
    ) {
      const size_t size = sizeof(array5<char>); // sizeof(T*) = sizeof(char*)
      hipError_t ierr   = hipMalloc((void**)device_copy,size);      
      if ( ierr == hipSuccess ) {
        return this->copy_self_to_device_async(*device_copy,stream);
      } else {
        return ierr;
      }
    }
    /**
     * Initialize.
     * \param[in] data_host host data pointer (may be nullptr; see the note).
     * \param[in] pinned the host array is pinned or should be pinned.
     * \param[in] copyout_at_destruction copy data out from the
     *            device to the host when this instance gets destroyed.
     * \note Allocates a new host_data pointer if the
     * passed pointer is a nullptr. If the pinned argument is true,
     * the new hostpotr is allocated via hipHostMalloc.
     * Otherwise, it is allocated via classic malloc.
     */
    __host__ hipError_t init(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int n1,int n2,int n3,int n4,int n5,
        int lb1,int lb2,int lb3,int lb4,int lb5,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      this->data.wrap(
          data_host,
          data_dev,
          n1,n2,n3,n4,n5,
          lb1,lb2,lb3,lb4,lb5
      );
      this->bytes_per_element = bytes_per_element;
      this->num_refs          = 1;
      this->alloc_mode        = alloc_mode;
      this->sync_mode         = sync_mode;
  
      // host array allocation
      hipError_t ierr = hipSuccess;
      switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
#endif
        case AllocMode::AllocPinnedHostWrapDevice:
        case AllocMode::AllocPinnedHostAllocDevice:
          ierr = hipHostMalloc((void**) &this->data.data_host,this->num_data_bytes(),0);
	  break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
          this->data.data_host = (T*) malloc(this->num_data_bytes());
	  break;
#endif
	case AllocMode::WrapHostWrapDevice:
        case AllocMode::WrapHostAllocDevice:
          // do nothing as already set by data.wrap call
	  break;
	default:
          std::cerr << "ERROR: gpufort::array5::init(...): Unexpected value for 'this->alloc_mode': " 
                    << static_cast<int>(this->alloc_mode) << std::endl; 
          std::terminate();
          break;
      }
      // device array allocation
      if ( ierr == hipSuccess ) {
        switch (this->alloc_mode) {
          case AllocMode::AllocHostWrapDevice:
          case AllocMode::AllocPinnedHostWrapDevice:
          case AllocMode::WrapHostWrapDevice:
            // do nothing as already set by data.wrap call
            break;
          case AllocMode::WrapHostAllocDevice:
          case AllocMode::AllocHostAllocDevice:
          case AllocMode::AllocPinnedHostAllocDevice:
            ierr = hipMalloc((void**) &this->data.data_dev, this->num_data_bytes());
            break;
          default:
            break;
        }
      }
      if ( ierr == hipSuccess ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::None:
          case SyncMode::Copyout:
          case SyncMode::InvertedCopyout:
            // do nothing
            break;
          case SyncMode::Copy:
          case SyncMode::Copyin:
            ierr = this->copy_to_device();
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyin:
            ierr = this->copy_to_host();
            break;
          default:
            std::cerr << "ERROR: gpufort::array5::init(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
      }
      return ierr;
    }
   
    __host__ hipError_t init(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int* sizes,
        int* lower_bounds,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      return this->init(
        bytes_per_element,
        data_host,
        data_dev,
        sizes[0],sizes[1],sizes[2],sizes[3],sizes[4],
        lower_bounds[0],lower_bounds[1],lower_bounds[2],lower_bounds[3],lower_bounds[4],alloc_mode,sync_mode);
    }
    
    /**
     * Destroy associated host and device data
     * if this is not a wrapper. Depending
     * on the pinned attribue, hipHostFree 
     * is used instead of free to deallocate
     * the host data.
     */
    __host__ hipError_t destroy() {
      hipError_t ierr = hipSuccess;
      if ( bytes_per_element > 0 ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::Copy:
          case SyncMode::Copyout:
            ierr = this->copy_to_host();
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyout:
            ierr = this->copy_to_device();
            break;
          case SyncMode::None:
          case SyncMode::Copyin:
          case SyncMode::InvertedCopyin:
            // do nothing
            break;
          default:
            std::cerr << "ERROR: gpufort::array5::destroy(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
        // host array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
#endif
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipHostFree(this->data.data_host);
              break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
              free(this->data.data_host); 
              break;
#endif
            case AllocMode::WrapHostWrapDevice:
            case AllocMode::WrapHostAllocDevice:
              break;
            default:
              std::cerr << "ERROR: gpufort::array5::destroy(...): Unexpected value for 'alloc_mode': " 
                        << static_cast<int>(alloc_mode) << std::endl; 
              std::terminate();
              break;
          }
        }
        // device array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::WrapHostWrapDevice:
              // do nothing as already set by data.wrap call
              break;
            case AllocMode::WrapHostAllocDevice:
            case AllocMode::AllocHostAllocDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipFree(this->data.data_dev);
              break;
            default:
              break;
          }
        }
        if ( ierr == hipSuccess ) {
          this->data.data_host    = nullptr;
          this->data.data_dev     = nullptr;
          this->bytes_per_element = -1; 
        } 
      }
      return ierr;
    }
    
    /**
     * Copy data to a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the destinaton buffer is interpreted as host or device variable
     * and the source is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_to_buffer(
       void* buffer, hipMemcpyKind memcpy_kind
    ) {
      T* src = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyDeviceToHost: 
        case hipMemcpyDeviceToDevice:
          src = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyHostToDevice:
          src = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array5::copy_to_buffer(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(src!=nullptr);
      #endif
      return hipMemcpy(
        buffer, 
        (void*) src,
        this->num_data_bytes(), 
        memcpy_kind);
    }
    
    /**
     * Copy data from a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the source buffer is interpreted as host or device variable
     * and the destination is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_from_buffer(
       void* buffer, hipMemcpyKind memcpy_kind
    ) {
      T* dest = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyHostToDevice: 
        case hipMemcpyDeviceToDevice:
          dest = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyDeviceToHost:
          dest = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array5::copy_from_buffer(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(dest!=nullptr);
      #endif
      return hipMemcpy(
        (void*) dest,
        buffer, 
        this->num_data_bytes(), 
        memcpy_kind);
    }
    
    /**
     * Copy host data to the device.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_host() {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpy(
        (void*) this->data.data_host, 
        (void*) this->data.data_dev,
        this->num_data_bytes(), 
        hipMemcpyDeviceToHost);
    }
    
    /**
     * Copy device data to the host.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_device() {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpy(
        (void*) this->data.data_dev, 
        (void*) this->data.data_host,
        this->num_data_bytes(),
        hipMemcpyHostToDevice);
    }
    
    /** 
     * Copies the struct to the device into an pre-allocated
     * device memory block but does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[in] device_struct device memory address to copy to
     */
    __host__ hipError_t copy_self_to_device(
        gpufort::array5<T>* device_struct
    ) {
      const size_t size = sizeof(array5<char>); // sizeof(T*) = sizeof(char*)
      return hipMemcpy(
          (void*) device_struct, 
          (void*) this,
          size,
          hipMemcpyHostToDevice);
    }

    /** 
     * Copies the struct to the device but
     * does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[inout] device_copy pointer to device copy pointer 
     */
    __host__ hipError_t create_device_copy(
        gpufort::array5<T>** device_copy
    ) {
      const size_t size = sizeof(array5<char>); // sizeof(T*) = sizeof(char*)
      hipError_t ierr   = hipMalloc((void**)device_copy,size);      
      if ( ierr == hipSuccess ) {
        return this->copy_self_to_device(*device_copy);
      } else {
        return ierr;
      }
    }

    /**
     * Allocate a host buffer with
     * the same size as the data buffers
     * associated with this gpufort array.
     * @see num_data_bytes()
     * \param[inout] pointer to the buffer to allocate
     * \param[in] pinned If the memory should be pinned (default=True)
     * \param[in] flags  Flags for the host memory allocation (default=0).
     */
    __host__ hipError_t allocate_host_buffer(void** buffer,bool pinned=true,int flags=0) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
      if ( pinned ) { 
        ierr = hipHostMalloc(buffer,this->num_data_bytes(),flags);
      } else {
        *buffer = malloc(this->num_data_bytes());
      }
      return ierr;
    }

    /**
     * Deallocate a host buffer
     * created via the allocate_host_buffer routine.
     * @see num_data_bytes(), allocate_host_buffer
     * \param[inout] the buffer to deallocte
     * \param[in] pinned If the memory to deallocate is pinned (default=True)
     */
    __host__ hipError_t deallocate_host_buffer(void* buffer,bool pinned=true) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
      if ( pinned ) { 
        ierr = hipHostFree(buffer);
      } else {
        free(buffer);
      }
      return ierr;
    }
    /**
     * Allocate a device buffer with
     * the same size as the data buffers
     * associated with this gpufort array.
     * @see num_data_bytes()
     * \param[inout] pointer to the buffer to allocate
     */
    __host__ hipError_t allocate_device_buffer(void** buffer) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
 
      ierr = hipMalloc(buffer,this->num_data_bytes());
      return ierr;
    }

    /**
     * Deallocate a device buffer
     * created via the allocate_device_buffer routine.
     * @see num_data_bytes(), allocate_device_buffer
     * \param[inout] the buffer to deallocte
     */
    __host__ hipError_t deallocate_device_buffer(void* buffer) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
 
      ierr = hipFree(buffer);
      return ierr;
    }

    /** 
     * Copy metadata from another gpufort array to this gpufort array.
     * \note The allocation mode is always set to AllocMode::WrapHostWrapDevice
     *       and the sync mode is set to SyncMode::None.
     * \note No deep copy, i.e. the host and device data is not copied, just the pointers
     * to the data.
     */
    __host__ void copy(const gpufort::array5<T>& other) {
      this->init(
        other.bytes_per_element,
        other.data.data_host,
        other.data.data_dev,
        other.size(1),other.size(2),other.size(3),other.size(4),other.size(5),
        other.lbound(1),other.lbound(2),other.lbound(3),other.lbound(4),other.lbound(5),
        AllocMode::WrapHostWrapDevice,
        SyncMode::None);
    }

    /** 
     *  \return Number of bytes required to store 
     *  the array elements on the host or the device.
     */ 
    __host__ size_t num_data_bytes() {
      return this->data.num_elements * this->bytes_per_element;
    }
    
    /**
     * \return the number of elements of this array.
     */
    __host__ int num_elements() {
      return this->data.num_elements;
    }

    /**
     * Linearize multi-dimensional index.
     *
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ int linearized_index (
      const int i1,
      const int i2,
      const int i3,
      const int i4,
      const int i5
    ) const {
      return this->data.linearized_index(
        i1,i2,i3,i4,i5
      );
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ T& operator() (
      const int i1,
      const int i2,
      const int i3,
      const int i4,
      const int i5
    ) {
      return this->data(
        i1,i2,i3,i4,i5
      );
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ const T& operator() (
      const int i1,
      const int i2,
      const int i3,
      const int i4,
      const int i5
    ) const {
      return this->data(
        i1,i2,i3,i4,i5
      );
    }
    
    /**
     * \return Size of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,5
     */
    __host__ __device__ __forceinline__ int size(int dim) const {
      return this->data.size(dim);
    }
    
    /**
     * \return Lower bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,5
     */
    __host__ __device__ __forceinline__ int lbound(int dim) const {
      return this->data.lbound(dim);
    }
    
    /**
     * \return Upper bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,5
     */
    __host__ __device__ __forceinline__ int ubound(int dim) const {
      return this->data.ubound(dim);
    }
    

    /**
     * Collapse the array by fixing 1 indices.
     * \return A gpufort array of rank 4.
     * \param[in] i5,...,i5 indices to fix.
     */
    __host__ gpufort::array4<T> collapse(
        const int i5
    ) const {
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1);
      const int n2  = this->size(2);
      const int lb2 = this->lbound(2);
      const int n3  = this->size(3);
      const int lb3 = this->lbound(3);
      const int n4  = this->size(4);
      const int lb4 = this->lbound(4);
      const int index = linearized_index(
        lb1,lb2,lb3,lb4,
        i5);
      T* data_host_new = nullptr; 
      if (this->data.data_host != nullptr) {
        data_host_new = this->data.data_host + index;
      }
      T* data_dev_new = nullptr; 
      if (this->data.data_dev != nullptr) {
        data_dev_new = this->data.data_dev + index;
      }
      gpufort::array4<T> collapsed_array;
      hipError_t ierr = 
        collapsed_array.init(
          this->bytes_per_element,
          data_host_new,
          data_dev_new,
          n1,n2,n3,n4,
          lb1,lb2,lb3,lb4,
          AllocMode::WrapHostWrapDevice,
          SyncMode::None);
      return collapsed_array;
    }
    /**
     * Collapse the array by fixing 2 indices.
     * \return A gpufort array of rank 3.
     * \param[in] i4,...,i5 indices to fix.
     */
    __host__ gpufort::array3<T> collapse(
        const int i4,
        const int i5
    ) const {
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1);
      const int n2  = this->size(2);
      const int lb2 = this->lbound(2);
      const int n3  = this->size(3);
      const int lb3 = this->lbound(3);
      const int index = linearized_index(
        lb1,lb2,lb3,
        i4,i5);
      T* data_host_new = nullptr; 
      if (this->data.data_host != nullptr) {
        data_host_new = this->data.data_host + index;
      }
      T* data_dev_new = nullptr; 
      if (this->data.data_dev != nullptr) {
        data_dev_new = this->data.data_dev + index;
      }
      gpufort::array3<T> collapsed_array;
      hipError_t ierr = 
        collapsed_array.init(
          this->bytes_per_element,
          data_host_new,
          data_dev_new,
          n1,n2,n3,
          lb1,lb2,lb3,
          AllocMode::WrapHostWrapDevice,
          SyncMode::None);
      return collapsed_array;
    }
    /**
     * Collapse the array by fixing 3 indices.
     * \return A gpufort array of rank 2.
     * \param[in] i3,...,i5 indices to fix.
     */
    __host__ gpufort::array2<T> collapse(
        const int i3,
        const int i4,
        const int i5
    ) const {
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1);
      const int n2  = this->size(2);
      const int lb2 = this->lbound(2);
      const int index = linearized_index(
        lb1,lb2,
        i3,i4,i5);
      T* data_host_new = nullptr; 
      if (this->data.data_host != nullptr) {
        data_host_new = this->data.data_host + index;
      }
      T* data_dev_new = nullptr; 
      if (this->data.data_dev != nullptr) {
        data_dev_new = this->data.data_dev + index;
      }
      gpufort::array2<T> collapsed_array;
      hipError_t ierr = 
        collapsed_array.init(
          this->bytes_per_element,
          data_host_new,
          data_dev_new,
          n1,n2,
          lb1,lb2,
          AllocMode::WrapHostWrapDevice,
          SyncMode::None);
      return collapsed_array;
    }
    /**
     * Collapse the array by fixing 4 indices.
     * \return A gpufort array of rank 1.
     * \param[in] i2,...,i5 indices to fix.
     */
    __host__ gpufort::array1<T> collapse(
        const int i2,
        const int i3,
        const int i4,
        const int i5
    ) const {
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1);
      const int index = linearized_index(
        lb1,
        i2,i3,i4,i5);
      T* data_host_new = nullptr; 
      if (this->data.data_host != nullptr) {
        data_host_new = this->data.data_host + index;
      }
      T* data_dev_new = nullptr; 
      if (this->data.data_dev != nullptr) {
        data_dev_new = this->data.data_dev + index;
      }
      gpufort::array1<T> collapsed_array;
      hipError_t ierr = 
        collapsed_array.init(
          this->bytes_per_element,
          data_host_new,
          data_dev_new,
          n1,
          lb1,
          AllocMode::WrapHostWrapDevice,
          SyncMode::None);
      return collapsed_array;
    }

    /**
     * Write host data values and their corresponding index
     * to an output stream.
     * \param[inout] out        output stream
     * \param[in]    prefix     prefix to put before each output line
     * \param[in]    print_prec precision to use when printing floating point numbers 
     *               [default=GPUFORT_ARRAY_PRINT_PREC[default=6]]
     * \note This method only makes sense for numeric data types and
     *       may result in compilation, runtime errors or undefined output if the
     *       underlying data is not numeric.
     */
    __host__ void print_host_data(
        std::ostream& out, const char* prefix, PrintMode print_mode,
        const int print_prec=GPUFORT_ARRAY_PRINT_PREC) const {
      const bool print_norms = 
	print_mode==PrintMode::PrintNorms ||
        print_mode==PrintMode::PrintValuesAndNorms ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      const bool print_values = 
        print_mode==PrintMode::PrintValues ||
        print_mode==PrintMode::PrintValuesAndNorms;
      const bool print_values_c_style = 
        print_mode==PrintMode::PrintCValues ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      if ( !print_norms && !print_values && !print_values_c_style ) {
        std::cerr << "ERROR: gpufort::array5::print_host_data(...): Unexpected value for 'print_mode': " 
                  << static_cast<int>(print_mode) << std::endl; 
        std::terminate();
      }
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1); 
      const int n2  = this->size(2);
      const int lb2 = this->lbound(2); 
      const int n3  = this->size(3);
      const int lb3 = this->lbound(3); 
      const int n4  = this->size(4);
      const int lb4 = this->lbound(4); 
      const int n5  = this->size(5);
      const int lb5 = this->lbound(5); 
      const int n = this->data.num_elements; 
      T* A_h   = this->data.data_host;
      T min = +std::numeric_limits<T>::max();
      T max = -std::numeric_limits<T>::max();
      T sum = 0;
      T l1  = 0;
      T l2  = 0;
      out << prefix << ":\n";
    for ( int i5 = 0; i5 < n5; i5++ ) {
    for ( int i4 = 0; i4 < n4; i4++ ) {
    for ( int i3 = 0; i3 < n3; i3++ ) {
    for ( int i2 = 0; i2 < n2; i2++ ) {
    for ( int i1 = 0; i1 < n1; i1++ ) {
      const int idx = this->linearized_index(i1,i2,i3,i4,i5);
      T value = A_h[idx];
      if ( print_norms ) {
        min  = std::min(value,min);
        max  = std::max(value,max);
        sum += value;
        l1  += std::abs(value);
        l2  += value*value;
      }
      if ( print_values ) {
          out << prefix << "(" << (lb1+i1) << "," << (lb2+i2) << "," << (lb3+i3) << "," << (lb4+i4) << "," << (lb5+i5) <<  ") = " << std::setprecision(print_prec) << value << "\n";
      } else if ( print_values_c_style ) {
          out << prefix << "[" << idx << "] = " << std::setprecision(print_prec) << value << "\n";
      }
    }}}}} // for loops
    if ( print_norms ) {
      out << prefix << ":min=" << std::setprecision(print_prec) << min << "\n";
      out << prefix << ":max=" << std::setprecision(print_prec) << max << "\n";
      out << prefix << ":sum=" << std::setprecision(print_prec) << sum << "\n";
      out << prefix << ":l1="  << std::setprecision(print_prec) << l1  << "\n";
      out << prefix << ":l2="  << std::setprecision(print_prec) << std::sqrt(l2) << "\n";
      out << prefix << "num_elements=" << std::setprecision(print_prec) << n << "\n";
    }
  }
    /**
     * Write device data values and their corresponding index
     * to an output stream.
     * \param[inout] out        output stream
     * \param[in]    prefix     prefix to put before each output line
     * \param[in]    print_prec precision to use when printing floating point numbers 
     *               [default=GPUFORT_ARRAY_PRINT_PREC[default=6]]
     * \note This method only makes sense for numeric data types and
     *       may result in compilation, runtime errors or undefined output if the
     *       underlying data is not numeric.
     */
    __host__ void print_device_data(
        std::ostream& out, const char* prefix, PrintMode print_mode,
        const int print_prec=GPUFORT_ARRAY_PRINT_PREC) const {
      const bool print_norms = 
	print_mode==PrintMode::PrintNorms ||
        print_mode==PrintMode::PrintValuesAndNorms ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      const bool print_values = 
        print_mode==PrintMode::PrintValues ||
        print_mode==PrintMode::PrintValuesAndNorms;
      const bool print_values_c_style = 
        print_mode==PrintMode::PrintCValues ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      if ( !print_norms && !print_values && !print_values_c_style ) {
        std::cerr << "ERROR: gpufort::array5::print_device_data(...): Unexpected value for 'print_mode': " 
                  << static_cast<int>(print_mode) << std::endl; 
        std::terminate();
      }
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1); 
      const int n2  = this->size(2);
      const int lb2 = this->lbound(2); 
      const int n3  = this->size(3);
      const int lb3 = this->lbound(3); 
      const int n4  = this->size(4);
      const int lb4 = this->lbound(4); 
      const int n5  = this->size(5);
      const int lb5 = this->lbound(5); 
      const int n = this->data.num_elements; 
      std::vector<T> host_array(n);
      T* A_h = host_array.data();
      T* A   = this->data.data_dev;
      HIP_CHECK(hipMemcpy(A_h, A, n*sizeof(T), hipMemcpyDeviceToHost));
      T min = +std::numeric_limits<T>::max();
      T max = -std::numeric_limits<T>::max();
      T sum = 0;
      T l1  = 0;
      T l2  = 0;
      out << prefix << ":\n";
    for ( int i5 = 0; i5 < n5; i5++ ) {
    for ( int i4 = 0; i4 < n4; i4++ ) {
    for ( int i3 = 0; i3 < n3; i3++ ) {
    for ( int i2 = 0; i2 < n2; i2++ ) {
    for ( int i1 = 0; i1 < n1; i1++ ) {
      const int idx = this->linearized_index(i1,i2,i3,i4,i5);
      T value = A_h[idx];
      if ( print_norms ) {
        min  = std::min(value,min);
        max  = std::max(value,max);
        sum += value;
        l1  += std::abs(value);
        l2  += value*value;
      }
      if ( print_values ) {
          out << prefix << "(" << (lb1+i1) << "," << (lb2+i2) << "," << (lb3+i3) << "," << (lb4+i4) << "," << (lb5+i5) <<  ") = " << std::setprecision(print_prec) << value << "\n";
      } else if ( print_values_c_style ) {
          out << prefix << "[" << idx << "] = " << std::setprecision(print_prec) << value << "\n";
      }
    }}}}} // for loops
    if ( print_norms ) {
      out << prefix << ":min=" << std::setprecision(print_prec) << min << "\n";
      out << prefix << ":max=" << std::setprecision(print_prec) << max << "\n";
      out << prefix << ":sum=" << std::setprecision(print_prec) << sum << "\n";
      out << prefix << ":l1="  << std::setprecision(print_prec) << l1  << "\n";
      out << prefix << ":l2="  << std::setprecision(print_prec) << std::sqrt(l2) << "\n";
      out << prefix << "num_elements=" << std::setprecision(print_prec) << n << "\n";
    }
  }

  };

  /**
   * Intended to be passed as kernel launch parameter.
   * \note Operator `T& operator()` requires parametrization with type T.
   * \note Size of this struct independent of template type T.
   */
  template<typename T>
  struct array_descr6 {
    T*     data_host = nullptr;
    T*     data_dev  = nullptr;
    int num_elements = 0;     //> Number of elements represented by this array.
    int index_offset = -1;    //> Offset for index calculation; scalar product of negative lower bounds and strides.
    int    stride2  = -1;    //> Stride 2 for linearizing 6-dimensional index.
    int    stride3  = -1;    //> Stride 3 for linearizing 6-dimensional index.
    int    stride4  = -1;    //> Stride 4 for linearizing 6-dimensional index.
    int    stride5  = -1;    //> Stride 5 for linearizing 6-dimensional index.
    int    stride6  = -1;    //> Stride 6 for linearizing 6-dimensional index.
 
    /**
     * Wrap a host-device pointer pair and store the associated
     * array meta data.
     *
     * \param[in] data_host host data pointer.
     * \param[in] data_dev device data pointer.
     * \param[in] n1,n2,... element counts per dimension of the multi-dim. array 
     * \param[in] lb1,lb2,... lower bounds, i.e. the index of the first element per dimension of the multi-dim. array 
     */ 
    __host__ __device__ void wrap(
        T* data_host,
        T* data_dev,
        const int n1,
        const int n2,
        const int n3,
        const int n4,
        const int n5,
        const int n6,
        const int lb1,
        const int lb2,
        const int lb3,
        const int lb4,
        const int lb5,
        const int lb6
      ) {
       this->data_host = data_host;
       this->data_dev  = data_dev;
       // column-major access
       this->stride2 = 1*n1;
       this->stride3 = this->stride2*n2;
       this->stride4 = this->stride3*n3;
       this->stride5 = this->stride4*n4;
       this->stride6 = this->stride5*n5;
       this->num_elements = this->stride6*n6;
       this->index_offset =
         -lb1*1
         -lb2*this->stride2
         -lb3*this->stride3
         -lb4*this->stride4
         -lb5*this->stride5
         -lb6*this->stride6;
    }
    
    /**
     * Linearize multi-dimensional index.
     *
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ int linearized_index (
      const int i1,
      const int i2,
      const int i3,
      const int i4,
      const int i5,
      const int i6
    ) const {
      return this->index_offset
          + i1*1
          + i2*this->stride2
          + i3*this->stride3
          + i4*this->stride4
          + i5*this->stride5
          + i6*this->stride6;
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ T& operator() (
      const int i1,
      const int i2,
      const int i3,
      const int i4,
      const int i5,
      const int i6
    ) {
      const int index = linearized_index(
        i1,i2,i3,i4,i5,i6
      );
      #ifdef __HIP_DEVICE_COMPILE__
      return this->data_dev[index];
      #else
      assert(i1 >= lbound(1));
      assert(i1 <= ubound(1));
      assert(i2 >= lbound(2));
      assert(i2 <= ubound(2));
      assert(i3 >= lbound(3));
      assert(i3 <= ubound(3));
      assert(i4 >= lbound(4));
      assert(i4 <= ubound(4));
      assert(i5 >= lbound(5));
      assert(i5 <= ubound(5));
      assert(i6 >= lbound(6));
      assert(i6 <= ubound(6));
      return this->data_host[index];
      #endif
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ const T& operator() (
      const int i1,
      const int i2,
      const int i3,
      const int i4,
      const int i5,
      const int i6
    ) const {
      const int index = linearized_index(
        i1,i2,i3,i4,i5,i6
      );
      #ifdef __HIP_DEVICE_COMPILE__
      return this->data_dev[index];
      #else
      assert(i1 >= lbound(1));
      assert(i1 <= ubound(1));
      assert(i2 >= lbound(2));
      assert(i2 <= ubound(2));
      assert(i3 >= lbound(3));
      assert(i3 <= ubound(3));
      assert(i4 >= lbound(4));
      assert(i4 <= ubound(4));
      assert(i5 >= lbound(5));
      assert(i5 <= ubound(5));
      assert(i6 >= lbound(6));
      assert(i6 <= ubound(6));
      return this->data_host[index];
      #endif
    }
    
    /**
     * \return Size of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,6
     */
    __host__ __device__ __forceinline__ int size(int dim) const {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(dim >= 1);
      assert(dim <= 6);
      #endif
      switch(dim) {
       case 6:
	        return this->num_elements / this->stride6;
       case 5:
	        return this->stride6 / this->stride5;
       case 4:
	        return this->stride5 / this->stride4;
       case 3:
	        return this->stride4 / this->stride3;
       case 2:
	        return this->stride3 / this->stride2;
       case 1:
	        return this->stride2 / 1;
       default:
          #ifndef __HIP_DEVICE_COMPILE__
          std::cerr << "‘dim’ argument of ‘gpufort::array6::size’ is not a valid dimension index ('dim': "<<dim<<", max dimension: 6" << std::endl;
          std::terminate();
          #else
          return -1;
          #endif
      }
    }
    
    /**
     * \return Lower bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,6
     */
    __host__ __device__ __forceinline__ int lbound(int dim) const {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(dim >= 1);
      assert(dim <= 6);
      #endif
      int offset_remainder = this->index_offset;
      int lb = -offset_remainder / this->stride6;
      if ( dim == 6 ) return lb;
      offset_remainder = offset_remainder + lb*this->stride6;
      lb = -offset_remainder / this->stride5;
      if ( dim == 5 ) return lb;
      offset_remainder = offset_remainder + lb*this->stride5;
      lb = -offset_remainder / this->stride4;
      if ( dim == 4 ) return lb;
      offset_remainder = offset_remainder + lb*this->stride4;
      lb = -offset_remainder / this->stride3;
      if ( dim == 3 ) return lb;
      offset_remainder = offset_remainder + lb*this->stride3;
      lb = -offset_remainder / this->stride2;
      if ( dim == 2 ) return lb;
      offset_remainder = offset_remainder + lb*this->stride2;
      lb = -offset_remainder / 1;
      #ifndef __HIP_DEVICE_COMPILE__
      offset_remainder = offset_remainder + lb*1;
      assert(offset_remainder == 0);
      #endif
      return lb;
    }
    
    /**
     * \return Upper bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,6
     */
    __host__ __device__ __forceinline__ int ubound(int dim) const {
      return this->lbound(dim) + this->size(dim) - 1;
    }
  };

  template<typename T>
  struct array6{
    array_descr6<T> data;
    AllocMode              alloc_mode = AllocMode::WrapHostAllocDevice; //> Data allocation strategy. Default: 
                                                                        //> wrap the host and allocate device data
    SyncMode               sync_mode  = SyncMode::None;                 //> How data should be synchronized
                                                                        //> during the initialization and destruction of this GPUFORT array.
    int num_refs             = 0;  //> Number of references.
    size_t bytes_per_element = -1; //> Bytes per element; stored to make num_data_bytes routine independent of T 

    array6() {
      // do nothing
    }
    
    ~array6() {
      // do nothing
    }

    /**
     * Initialize.
     * \param[in] data_host host data pointer (may be nullptr; see the note).
     * \param[in] pinned the host array is pinned or should be pinned.
     * \param[in] copyout_at_destruction copy data out from the
     *            device to the host when this instance gets destroyed.
     * \note Allocates a new host_data pointer if the
     * passed pointer is a nullptr. If the pinned argument is true,
     * the new hostpotr is allocated via hipHostMalloc.
     * Otherwise, it is allocated via classic malloc.
     */
    __host__ hipError_t init_async(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int n1,int n2,int n3,int n4,int n5,int n6,
        int lb1,int lb2,int lb3,int lb4,int lb5,int lb6,
        hipStream_t stream,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      this->data.wrap(
          data_host,
          data_dev,
          n1,n2,n3,n4,n5,n6,
          lb1,lb2,lb3,lb4,lb5,lb6
      );
      this->bytes_per_element = bytes_per_element;
      this->num_refs          = 1;
      this->alloc_mode        = alloc_mode;
      this->sync_mode         = sync_mode;
  
      // host array allocation
      hipError_t ierr = hipSuccess;
      switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
#endif
        case AllocMode::AllocPinnedHostWrapDevice:
        case AllocMode::AllocPinnedHostAllocDevice:
          ierr = hipHostMalloc((void**) &this->data.data_host,this->num_data_bytes(),0);
	  break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
          this->data.data_host = (T*) malloc(this->num_data_bytes());
	  break;
#endif
	case AllocMode::WrapHostWrapDevice:
        case AllocMode::WrapHostAllocDevice:
          // do nothing as already set by data.wrap call
	  break;
	default:
          std::cerr << "ERROR: gpufort::array6::init_async(...): Unexpected value for 'this->alloc_mode': " 
                    << static_cast<int>(this->alloc_mode) << std::endl; 
          std::terminate();
          break;
      }
      // device array allocation
      if ( ierr == hipSuccess ) {
        switch (this->alloc_mode) {
          case AllocMode::AllocHostWrapDevice:
          case AllocMode::AllocPinnedHostWrapDevice:
          case AllocMode::WrapHostWrapDevice:
            // do nothing as already set by data.wrap call
            break;
          case AllocMode::WrapHostAllocDevice:
          case AllocMode::AllocHostAllocDevice:
          case AllocMode::AllocPinnedHostAllocDevice:
            ierr = hipMalloc((void**) &this->data.data_dev, this->num_data_bytes());
            break;
          default:
            break;
        }
      }
      if ( ierr == hipSuccess ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::None:
          case SyncMode::Copyout:
          case SyncMode::InvertedCopyout:
            // do nothing
            break;
          case SyncMode::Copy:
          case SyncMode::Copyin:
            ierr = this->copy_to_device_async(stream);
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyin:
            ierr = this->copy_to_host_async(stream);
            break;
          default:
            std::cerr << "ERROR: gpufort::array6::init_async(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
      }
      return ierr;
    }
   
    __host__ hipError_t init_async(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int* sizes,
        int* lower_bounds,
        hipStream_t stream,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      return this->init_async(
        bytes_per_element,
        data_host,
        data_dev,
        sizes[0],sizes[1],sizes[2],sizes[3],sizes[4],sizes[5],
        lower_bounds[0],lower_bounds[1],lower_bounds[2],lower_bounds[3],lower_bounds[4],lower_bounds[5],
        stream,alloc_mode,sync_mode);
    }
    
    /**
     * Destroy associated host and device data
     * if this is not a wrapper. Depending
     * on the pinned attribue, hipHostFree 
     * is used instead of free to deallocate
     * the host data.
     */
    __host__ hipError_t destroy_async(hipStream_t stream) {
      hipError_t ierr = hipSuccess;
      if ( bytes_per_element > 0 ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::Copy:
          case SyncMode::Copyout:
            ierr = this->copy_to_host_async(stream);
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyout:
            ierr = this->copy_to_device_async(stream);
            break;
          case SyncMode::None:
          case SyncMode::Copyin:
          case SyncMode::InvertedCopyin:
            // do nothing
            break;
          default:
            std::cerr << "ERROR: gpufort::array6::destroy_async(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
        // host array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
#endif
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipHostFree(this->data.data_host);
              break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
              free(this->data.data_host); 
              break;
#endif
            case AllocMode::WrapHostWrapDevice:
            case AllocMode::WrapHostAllocDevice:
              break;
            default:
              std::cerr << "ERROR: gpufort::array6::destroy_async(...): Unexpected value for 'alloc_mode': " 
                        << static_cast<int>(alloc_mode) << std::endl; 
              std::terminate();
              break;
          }
        }
        // device array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::WrapHostWrapDevice:
              // do nothing as already set by data.wrap call
              break;
            case AllocMode::WrapHostAllocDevice:
            case AllocMode::AllocHostAllocDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipFree(this->data.data_dev);
              break;
            default:
              break;
          }
        }
        if ( ierr == hipSuccess ) {
          this->data.data_host    = nullptr;
          this->data.data_dev     = nullptr;
          this->bytes_per_element = -1; 
        } 
      }
      return ierr;
    }
    
    /**
     * Copy data to a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the destinaton buffer is interpreted as host or device variable
     * and the source is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_to_buffer_async(
       void* buffer, hipMemcpyKind memcpy_kind,
       hipStream_t stream
    ) {
      T* src = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyDeviceToHost: 
        case hipMemcpyDeviceToDevice:
          src = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyHostToDevice:
          src = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array6::copy_to_buffer_async(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(src!=nullptr);
      #endif
      return hipMemcpyAsync(
        buffer, 
        (void*) src,
        this->num_data_bytes(), 
        memcpy_kind, stream);
    }
    
    /**
     * Copy data from a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the source buffer is interpreted as host or device variable
     * and the destination is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_from_buffer_async(
       void* buffer, hipMemcpyKind memcpy_kind,
       hipStream_t stream
    ) {
      T* dest = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyHostToDevice: 
        case hipMemcpyDeviceToDevice:
          dest = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyDeviceToHost:
          dest = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array6::copy_from_buffer_async(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(dest!=nullptr);
      #endif
      return hipMemcpyAsync(
        (void*) dest,
        buffer, 
        this->num_data_bytes(), 
        memcpy_kind, stream);
    }
    
    /**
     * Copy host data to the device.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_host_async(hipStream_t stream) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpyAsync(
        (void*) this->data.data_host, 
        (void*) this->data.data_dev,
        this->num_data_bytes(), 
        hipMemcpyDeviceToHost, stream);
    }
    
    /**
     * Copy device data to the host.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_device_async(hipStream_t stream) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpyAsync(
        (void*) this->data.data_dev, 
        (void*) this->data.data_host,
        this->num_data_bytes(),
        hipMemcpyHostToDevice, stream);
    }
    
    /** 
     * Copies the struct to the device into an pre-allocated
     * device memory block but does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[in] device_struct device memory address to copy to
     */
    __host__ hipError_t copy_self_to_device_async(
        gpufort::array6<T>* device_struct,
        hipStream_t stream
    ) {
      const size_t size = sizeof(array6<char>); // sizeof(T*) = sizeof(char*)
      return hipMemcpyAsync(
          (void*) device_struct, 
          (void*) this,
          size,
          hipMemcpyHostToDevice, stream);
    }

    /** 
     * Copies the struct to the device but
     * does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[inout] device_copy pointer to device copy pointer 
     */
    __host__ hipError_t create_device_copy_async(
        gpufort::array6<T>** device_copy,
        hipStream_t stream
    ) {
      const size_t size = sizeof(array6<char>); // sizeof(T*) = sizeof(char*)
      hipError_t ierr   = hipMalloc((void**)device_copy,size);      
      if ( ierr == hipSuccess ) {
        return this->copy_self_to_device_async(*device_copy,stream);
      } else {
        return ierr;
      }
    }
    /**
     * Initialize.
     * \param[in] data_host host data pointer (may be nullptr; see the note).
     * \param[in] pinned the host array is pinned or should be pinned.
     * \param[in] copyout_at_destruction copy data out from the
     *            device to the host when this instance gets destroyed.
     * \note Allocates a new host_data pointer if the
     * passed pointer is a nullptr. If the pinned argument is true,
     * the new hostpotr is allocated via hipHostMalloc.
     * Otherwise, it is allocated via classic malloc.
     */
    __host__ hipError_t init(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int n1,int n2,int n3,int n4,int n5,int n6,
        int lb1,int lb2,int lb3,int lb4,int lb5,int lb6,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      this->data.wrap(
          data_host,
          data_dev,
          n1,n2,n3,n4,n5,n6,
          lb1,lb2,lb3,lb4,lb5,lb6
      );
      this->bytes_per_element = bytes_per_element;
      this->num_refs          = 1;
      this->alloc_mode        = alloc_mode;
      this->sync_mode         = sync_mode;
  
      // host array allocation
      hipError_t ierr = hipSuccess;
      switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
#endif
        case AllocMode::AllocPinnedHostWrapDevice:
        case AllocMode::AllocPinnedHostAllocDevice:
          ierr = hipHostMalloc((void**) &this->data.data_host,this->num_data_bytes(),0);
	  break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
          this->data.data_host = (T*) malloc(this->num_data_bytes());
	  break;
#endif
	case AllocMode::WrapHostWrapDevice:
        case AllocMode::WrapHostAllocDevice:
          // do nothing as already set by data.wrap call
	  break;
	default:
          std::cerr << "ERROR: gpufort::array6::init(...): Unexpected value for 'this->alloc_mode': " 
                    << static_cast<int>(this->alloc_mode) << std::endl; 
          std::terminate();
          break;
      }
      // device array allocation
      if ( ierr == hipSuccess ) {
        switch (this->alloc_mode) {
          case AllocMode::AllocHostWrapDevice:
          case AllocMode::AllocPinnedHostWrapDevice:
          case AllocMode::WrapHostWrapDevice:
            // do nothing as already set by data.wrap call
            break;
          case AllocMode::WrapHostAllocDevice:
          case AllocMode::AllocHostAllocDevice:
          case AllocMode::AllocPinnedHostAllocDevice:
            ierr = hipMalloc((void**) &this->data.data_dev, this->num_data_bytes());
            break;
          default:
            break;
        }
      }
      if ( ierr == hipSuccess ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::None:
          case SyncMode::Copyout:
          case SyncMode::InvertedCopyout:
            // do nothing
            break;
          case SyncMode::Copy:
          case SyncMode::Copyin:
            ierr = this->copy_to_device();
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyin:
            ierr = this->copy_to_host();
            break;
          default:
            std::cerr << "ERROR: gpufort::array6::init(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
      }
      return ierr;
    }
   
    __host__ hipError_t init(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int* sizes,
        int* lower_bounds,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      return this->init(
        bytes_per_element,
        data_host,
        data_dev,
        sizes[0],sizes[1],sizes[2],sizes[3],sizes[4],sizes[5],
        lower_bounds[0],lower_bounds[1],lower_bounds[2],lower_bounds[3],lower_bounds[4],lower_bounds[5],alloc_mode,sync_mode);
    }
    
    /**
     * Destroy associated host and device data
     * if this is not a wrapper. Depending
     * on the pinned attribue, hipHostFree 
     * is used instead of free to deallocate
     * the host data.
     */
    __host__ hipError_t destroy() {
      hipError_t ierr = hipSuccess;
      if ( bytes_per_element > 0 ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::Copy:
          case SyncMode::Copyout:
            ierr = this->copy_to_host();
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyout:
            ierr = this->copy_to_device();
            break;
          case SyncMode::None:
          case SyncMode::Copyin:
          case SyncMode::InvertedCopyin:
            // do nothing
            break;
          default:
            std::cerr << "ERROR: gpufort::array6::destroy(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
        // host array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
#endif
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipHostFree(this->data.data_host);
              break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
              free(this->data.data_host); 
              break;
#endif
            case AllocMode::WrapHostWrapDevice:
            case AllocMode::WrapHostAllocDevice:
              break;
            default:
              std::cerr << "ERROR: gpufort::array6::destroy(...): Unexpected value for 'alloc_mode': " 
                        << static_cast<int>(alloc_mode) << std::endl; 
              std::terminate();
              break;
          }
        }
        // device array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::WrapHostWrapDevice:
              // do nothing as already set by data.wrap call
              break;
            case AllocMode::WrapHostAllocDevice:
            case AllocMode::AllocHostAllocDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipFree(this->data.data_dev);
              break;
            default:
              break;
          }
        }
        if ( ierr == hipSuccess ) {
          this->data.data_host    = nullptr;
          this->data.data_dev     = nullptr;
          this->bytes_per_element = -1; 
        } 
      }
      return ierr;
    }
    
    /**
     * Copy data to a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the destinaton buffer is interpreted as host or device variable
     * and the source is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_to_buffer(
       void* buffer, hipMemcpyKind memcpy_kind
    ) {
      T* src = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyDeviceToHost: 
        case hipMemcpyDeviceToDevice:
          src = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyHostToDevice:
          src = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array6::copy_to_buffer(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(src!=nullptr);
      #endif
      return hipMemcpy(
        buffer, 
        (void*) src,
        this->num_data_bytes(), 
        memcpy_kind);
    }
    
    /**
     * Copy data from a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the source buffer is interpreted as host or device variable
     * and the destination is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_from_buffer(
       void* buffer, hipMemcpyKind memcpy_kind
    ) {
      T* dest = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyHostToDevice: 
        case hipMemcpyDeviceToDevice:
          dest = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyDeviceToHost:
          dest = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array6::copy_from_buffer(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(dest!=nullptr);
      #endif
      return hipMemcpy(
        (void*) dest,
        buffer, 
        this->num_data_bytes(), 
        memcpy_kind);
    }
    
    /**
     * Copy host data to the device.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_host() {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpy(
        (void*) this->data.data_host, 
        (void*) this->data.data_dev,
        this->num_data_bytes(), 
        hipMemcpyDeviceToHost);
    }
    
    /**
     * Copy device data to the host.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_device() {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpy(
        (void*) this->data.data_dev, 
        (void*) this->data.data_host,
        this->num_data_bytes(),
        hipMemcpyHostToDevice);
    }
    
    /** 
     * Copies the struct to the device into an pre-allocated
     * device memory block but does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[in] device_struct device memory address to copy to
     */
    __host__ hipError_t copy_self_to_device(
        gpufort::array6<T>* device_struct
    ) {
      const size_t size = sizeof(array6<char>); // sizeof(T*) = sizeof(char*)
      return hipMemcpy(
          (void*) device_struct, 
          (void*) this,
          size,
          hipMemcpyHostToDevice);
    }

    /** 
     * Copies the struct to the device but
     * does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[inout] device_copy pointer to device copy pointer 
     */
    __host__ hipError_t create_device_copy(
        gpufort::array6<T>** device_copy
    ) {
      const size_t size = sizeof(array6<char>); // sizeof(T*) = sizeof(char*)
      hipError_t ierr   = hipMalloc((void**)device_copy,size);      
      if ( ierr == hipSuccess ) {
        return this->copy_self_to_device(*device_copy);
      } else {
        return ierr;
      }
    }

    /**
     * Allocate a host buffer with
     * the same size as the data buffers
     * associated with this gpufort array.
     * @see num_data_bytes()
     * \param[inout] pointer to the buffer to allocate
     * \param[in] pinned If the memory should be pinned (default=True)
     * \param[in] flags  Flags for the host memory allocation (default=0).
     */
    __host__ hipError_t allocate_host_buffer(void** buffer,bool pinned=true,int flags=0) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
      if ( pinned ) { 
        ierr = hipHostMalloc(buffer,this->num_data_bytes(),flags);
      } else {
        *buffer = malloc(this->num_data_bytes());
      }
      return ierr;
    }

    /**
     * Deallocate a host buffer
     * created via the allocate_host_buffer routine.
     * @see num_data_bytes(), allocate_host_buffer
     * \param[inout] the buffer to deallocte
     * \param[in] pinned If the memory to deallocate is pinned (default=True)
     */
    __host__ hipError_t deallocate_host_buffer(void* buffer,bool pinned=true) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
      if ( pinned ) { 
        ierr = hipHostFree(buffer);
      } else {
        free(buffer);
      }
      return ierr;
    }
    /**
     * Allocate a device buffer with
     * the same size as the data buffers
     * associated with this gpufort array.
     * @see num_data_bytes()
     * \param[inout] pointer to the buffer to allocate
     */
    __host__ hipError_t allocate_device_buffer(void** buffer) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
 
      ierr = hipMalloc(buffer,this->num_data_bytes());
      return ierr;
    }

    /**
     * Deallocate a device buffer
     * created via the allocate_device_buffer routine.
     * @see num_data_bytes(), allocate_device_buffer
     * \param[inout] the buffer to deallocte
     */
    __host__ hipError_t deallocate_device_buffer(void* buffer) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
 
      ierr = hipFree(buffer);
      return ierr;
    }

    /** 
     * Copy metadata from another gpufort array to this gpufort array.
     * \note The allocation mode is always set to AllocMode::WrapHostWrapDevice
     *       and the sync mode is set to SyncMode::None.
     * \note No deep copy, i.e. the host and device data is not copied, just the pointers
     * to the data.
     */
    __host__ void copy(const gpufort::array6<T>& other) {
      this->init(
        other.bytes_per_element,
        other.data.data_host,
        other.data.data_dev,
        other.size(1),other.size(2),other.size(3),other.size(4),other.size(5),other.size(6),
        other.lbound(1),other.lbound(2),other.lbound(3),other.lbound(4),other.lbound(5),other.lbound(6),
        AllocMode::WrapHostWrapDevice,
        SyncMode::None);
    }

    /** 
     *  \return Number of bytes required to store 
     *  the array elements on the host or the device.
     */ 
    __host__ size_t num_data_bytes() {
      return this->data.num_elements * this->bytes_per_element;
    }
    
    /**
     * \return the number of elements of this array.
     */
    __host__ int num_elements() {
      return this->data.num_elements;
    }

    /**
     * Linearize multi-dimensional index.
     *
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ int linearized_index (
      const int i1,
      const int i2,
      const int i3,
      const int i4,
      const int i5,
      const int i6
    ) const {
      return this->data.linearized_index(
        i1,i2,i3,i4,i5,i6
      );
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ T& operator() (
      const int i1,
      const int i2,
      const int i3,
      const int i4,
      const int i5,
      const int i6
    ) {
      return this->data(
        i1,i2,i3,i4,i5,i6
      );
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ const T& operator() (
      const int i1,
      const int i2,
      const int i3,
      const int i4,
      const int i5,
      const int i6
    ) const {
      return this->data(
        i1,i2,i3,i4,i5,i6
      );
    }
    
    /**
     * \return Size of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,6
     */
    __host__ __device__ __forceinline__ int size(int dim) const {
      return this->data.size(dim);
    }
    
    /**
     * \return Lower bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,6
     */
    __host__ __device__ __forceinline__ int lbound(int dim) const {
      return this->data.lbound(dim);
    }
    
    /**
     * \return Upper bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,6
     */
    __host__ __device__ __forceinline__ int ubound(int dim) const {
      return this->data.ubound(dim);
    }
    

    /**
     * Collapse the array by fixing 1 indices.
     * \return A gpufort array of rank 5.
     * \param[in] i6,...,i6 indices to fix.
     */
    __host__ gpufort::array5<T> collapse(
        const int i6
    ) const {
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1);
      const int n2  = this->size(2);
      const int lb2 = this->lbound(2);
      const int n3  = this->size(3);
      const int lb3 = this->lbound(3);
      const int n4  = this->size(4);
      const int lb4 = this->lbound(4);
      const int n5  = this->size(5);
      const int lb5 = this->lbound(5);
      const int index = linearized_index(
        lb1,lb2,lb3,lb4,lb5,
        i6);
      T* data_host_new = nullptr; 
      if (this->data.data_host != nullptr) {
        data_host_new = this->data.data_host + index;
      }
      T* data_dev_new = nullptr; 
      if (this->data.data_dev != nullptr) {
        data_dev_new = this->data.data_dev + index;
      }
      gpufort::array5<T> collapsed_array;
      hipError_t ierr = 
        collapsed_array.init(
          this->bytes_per_element,
          data_host_new,
          data_dev_new,
          n1,n2,n3,n4,n5,
          lb1,lb2,lb3,lb4,lb5,
          AllocMode::WrapHostWrapDevice,
          SyncMode::None);
      return collapsed_array;
    }
    /**
     * Collapse the array by fixing 2 indices.
     * \return A gpufort array of rank 4.
     * \param[in] i5,...,i6 indices to fix.
     */
    __host__ gpufort::array4<T> collapse(
        const int i5,
        const int i6
    ) const {
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1);
      const int n2  = this->size(2);
      const int lb2 = this->lbound(2);
      const int n3  = this->size(3);
      const int lb3 = this->lbound(3);
      const int n4  = this->size(4);
      const int lb4 = this->lbound(4);
      const int index = linearized_index(
        lb1,lb2,lb3,lb4,
        i5,i6);
      T* data_host_new = nullptr; 
      if (this->data.data_host != nullptr) {
        data_host_new = this->data.data_host + index;
      }
      T* data_dev_new = nullptr; 
      if (this->data.data_dev != nullptr) {
        data_dev_new = this->data.data_dev + index;
      }
      gpufort::array4<T> collapsed_array;
      hipError_t ierr = 
        collapsed_array.init(
          this->bytes_per_element,
          data_host_new,
          data_dev_new,
          n1,n2,n3,n4,
          lb1,lb2,lb3,lb4,
          AllocMode::WrapHostWrapDevice,
          SyncMode::None);
      return collapsed_array;
    }
    /**
     * Collapse the array by fixing 3 indices.
     * \return A gpufort array of rank 3.
     * \param[in] i4,...,i6 indices to fix.
     */
    __host__ gpufort::array3<T> collapse(
        const int i4,
        const int i5,
        const int i6
    ) const {
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1);
      const int n2  = this->size(2);
      const int lb2 = this->lbound(2);
      const int n3  = this->size(3);
      const int lb3 = this->lbound(3);
      const int index = linearized_index(
        lb1,lb2,lb3,
        i4,i5,i6);
      T* data_host_new = nullptr; 
      if (this->data.data_host != nullptr) {
        data_host_new = this->data.data_host + index;
      }
      T* data_dev_new = nullptr; 
      if (this->data.data_dev != nullptr) {
        data_dev_new = this->data.data_dev + index;
      }
      gpufort::array3<T> collapsed_array;
      hipError_t ierr = 
        collapsed_array.init(
          this->bytes_per_element,
          data_host_new,
          data_dev_new,
          n1,n2,n3,
          lb1,lb2,lb3,
          AllocMode::WrapHostWrapDevice,
          SyncMode::None);
      return collapsed_array;
    }
    /**
     * Collapse the array by fixing 4 indices.
     * \return A gpufort array of rank 2.
     * \param[in] i3,...,i6 indices to fix.
     */
    __host__ gpufort::array2<T> collapse(
        const int i3,
        const int i4,
        const int i5,
        const int i6
    ) const {
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1);
      const int n2  = this->size(2);
      const int lb2 = this->lbound(2);
      const int index = linearized_index(
        lb1,lb2,
        i3,i4,i5,i6);
      T* data_host_new = nullptr; 
      if (this->data.data_host != nullptr) {
        data_host_new = this->data.data_host + index;
      }
      T* data_dev_new = nullptr; 
      if (this->data.data_dev != nullptr) {
        data_dev_new = this->data.data_dev + index;
      }
      gpufort::array2<T> collapsed_array;
      hipError_t ierr = 
        collapsed_array.init(
          this->bytes_per_element,
          data_host_new,
          data_dev_new,
          n1,n2,
          lb1,lb2,
          AllocMode::WrapHostWrapDevice,
          SyncMode::None);
      return collapsed_array;
    }
    /**
     * Collapse the array by fixing 5 indices.
     * \return A gpufort array of rank 1.
     * \param[in] i2,...,i6 indices to fix.
     */
    __host__ gpufort::array1<T> collapse(
        const int i2,
        const int i3,
        const int i4,
        const int i5,
        const int i6
    ) const {
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1);
      const int index = linearized_index(
        lb1,
        i2,i3,i4,i5,i6);
      T* data_host_new = nullptr; 
      if (this->data.data_host != nullptr) {
        data_host_new = this->data.data_host + index;
      }
      T* data_dev_new = nullptr; 
      if (this->data.data_dev != nullptr) {
        data_dev_new = this->data.data_dev + index;
      }
      gpufort::array1<T> collapsed_array;
      hipError_t ierr = 
        collapsed_array.init(
          this->bytes_per_element,
          data_host_new,
          data_dev_new,
          n1,
          lb1,
          AllocMode::WrapHostWrapDevice,
          SyncMode::None);
      return collapsed_array;
    }

    /**
     * Write host data values and their corresponding index
     * to an output stream.
     * \param[inout] out        output stream
     * \param[in]    prefix     prefix to put before each output line
     * \param[in]    print_prec precision to use when printing floating point numbers 
     *               [default=GPUFORT_ARRAY_PRINT_PREC[default=6]]
     * \note This method only makes sense for numeric data types and
     *       may result in compilation, runtime errors or undefined output if the
     *       underlying data is not numeric.
     */
    __host__ void print_host_data(
        std::ostream& out, const char* prefix, PrintMode print_mode,
        const int print_prec=GPUFORT_ARRAY_PRINT_PREC) const {
      const bool print_norms = 
	print_mode==PrintMode::PrintNorms ||
        print_mode==PrintMode::PrintValuesAndNorms ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      const bool print_values = 
        print_mode==PrintMode::PrintValues ||
        print_mode==PrintMode::PrintValuesAndNorms;
      const bool print_values_c_style = 
        print_mode==PrintMode::PrintCValues ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      if ( !print_norms && !print_values && !print_values_c_style ) {
        std::cerr << "ERROR: gpufort::array6::print_host_data(...): Unexpected value for 'print_mode': " 
                  << static_cast<int>(print_mode) << std::endl; 
        std::terminate();
      }
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1); 
      const int n2  = this->size(2);
      const int lb2 = this->lbound(2); 
      const int n3  = this->size(3);
      const int lb3 = this->lbound(3); 
      const int n4  = this->size(4);
      const int lb4 = this->lbound(4); 
      const int n5  = this->size(5);
      const int lb5 = this->lbound(5); 
      const int n6  = this->size(6);
      const int lb6 = this->lbound(6); 
      const int n = this->data.num_elements; 
      T* A_h   = this->data.data_host;
      T min = +std::numeric_limits<T>::max();
      T max = -std::numeric_limits<T>::max();
      T sum = 0;
      T l1  = 0;
      T l2  = 0;
      out << prefix << ":\n";
    for ( int i6 = 0; i6 < n6; i6++ ) {
    for ( int i5 = 0; i5 < n5; i5++ ) {
    for ( int i4 = 0; i4 < n4; i4++ ) {
    for ( int i3 = 0; i3 < n3; i3++ ) {
    for ( int i2 = 0; i2 < n2; i2++ ) {
    for ( int i1 = 0; i1 < n1; i1++ ) {
      const int idx = this->linearized_index(i1,i2,i3,i4,i5,i6);
      T value = A_h[idx];
      if ( print_norms ) {
        min  = std::min(value,min);
        max  = std::max(value,max);
        sum += value;
        l1  += std::abs(value);
        l2  += value*value;
      }
      if ( print_values ) {
          out << prefix << "(" << (lb1+i1) << "," << (lb2+i2) << "," << (lb3+i3) << "," << (lb4+i4) << "," << (lb5+i5) << "," << (lb6+i6) <<  ") = " << std::setprecision(print_prec) << value << "\n";
      } else if ( print_values_c_style ) {
          out << prefix << "[" << idx << "] = " << std::setprecision(print_prec) << value << "\n";
      }
    }}}}}} // for loops
    if ( print_norms ) {
      out << prefix << ":min=" << std::setprecision(print_prec) << min << "\n";
      out << prefix << ":max=" << std::setprecision(print_prec) << max << "\n";
      out << prefix << ":sum=" << std::setprecision(print_prec) << sum << "\n";
      out << prefix << ":l1="  << std::setprecision(print_prec) << l1  << "\n";
      out << prefix << ":l2="  << std::setprecision(print_prec) << std::sqrt(l2) << "\n";
      out << prefix << "num_elements=" << std::setprecision(print_prec) << n << "\n";
    }
  }
    /**
     * Write device data values and their corresponding index
     * to an output stream.
     * \param[inout] out        output stream
     * \param[in]    prefix     prefix to put before each output line
     * \param[in]    print_prec precision to use when printing floating point numbers 
     *               [default=GPUFORT_ARRAY_PRINT_PREC[default=6]]
     * \note This method only makes sense for numeric data types and
     *       may result in compilation, runtime errors or undefined output if the
     *       underlying data is not numeric.
     */
    __host__ void print_device_data(
        std::ostream& out, const char* prefix, PrintMode print_mode,
        const int print_prec=GPUFORT_ARRAY_PRINT_PREC) const {
      const bool print_norms = 
	print_mode==PrintMode::PrintNorms ||
        print_mode==PrintMode::PrintValuesAndNorms ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      const bool print_values = 
        print_mode==PrintMode::PrintValues ||
        print_mode==PrintMode::PrintValuesAndNorms;
      const bool print_values_c_style = 
        print_mode==PrintMode::PrintCValues ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      if ( !print_norms && !print_values && !print_values_c_style ) {
        std::cerr << "ERROR: gpufort::array6::print_device_data(...): Unexpected value for 'print_mode': " 
                  << static_cast<int>(print_mode) << std::endl; 
        std::terminate();
      }
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1); 
      const int n2  = this->size(2);
      const int lb2 = this->lbound(2); 
      const int n3  = this->size(3);
      const int lb3 = this->lbound(3); 
      const int n4  = this->size(4);
      const int lb4 = this->lbound(4); 
      const int n5  = this->size(5);
      const int lb5 = this->lbound(5); 
      const int n6  = this->size(6);
      const int lb6 = this->lbound(6); 
      const int n = this->data.num_elements; 
      std::vector<T> host_array(n);
      T* A_h = host_array.data();
      T* A   = this->data.data_dev;
      HIP_CHECK(hipMemcpy(A_h, A, n*sizeof(T), hipMemcpyDeviceToHost));
      T min = +std::numeric_limits<T>::max();
      T max = -std::numeric_limits<T>::max();
      T sum = 0;
      T l1  = 0;
      T l2  = 0;
      out << prefix << ":\n";
    for ( int i6 = 0; i6 < n6; i6++ ) {
    for ( int i5 = 0; i5 < n5; i5++ ) {
    for ( int i4 = 0; i4 < n4; i4++ ) {
    for ( int i3 = 0; i3 < n3; i3++ ) {
    for ( int i2 = 0; i2 < n2; i2++ ) {
    for ( int i1 = 0; i1 < n1; i1++ ) {
      const int idx = this->linearized_index(i1,i2,i3,i4,i5,i6);
      T value = A_h[idx];
      if ( print_norms ) {
        min  = std::min(value,min);
        max  = std::max(value,max);
        sum += value;
        l1  += std::abs(value);
        l2  += value*value;
      }
      if ( print_values ) {
          out << prefix << "(" << (lb1+i1) << "," << (lb2+i2) << "," << (lb3+i3) << "," << (lb4+i4) << "," << (lb5+i5) << "," << (lb6+i6) <<  ") = " << std::setprecision(print_prec) << value << "\n";
      } else if ( print_values_c_style ) {
          out << prefix << "[" << idx << "] = " << std::setprecision(print_prec) << value << "\n";
      }
    }}}}}} // for loops
    if ( print_norms ) {
      out << prefix << ":min=" << std::setprecision(print_prec) << min << "\n";
      out << prefix << ":max=" << std::setprecision(print_prec) << max << "\n";
      out << prefix << ":sum=" << std::setprecision(print_prec) << sum << "\n";
      out << prefix << ":l1="  << std::setprecision(print_prec) << l1  << "\n";
      out << prefix << ":l2="  << std::setprecision(print_prec) << std::sqrt(l2) << "\n";
      out << prefix << "num_elements=" << std::setprecision(print_prec) << n << "\n";
    }
  }

  };

  /**
   * Intended to be passed as kernel launch parameter.
   * \note Operator `T& operator()` requires parametrization with type T.
   * \note Size of this struct independent of template type T.
   */
  template<typename T>
  struct array_descr7 {
    T*     data_host = nullptr;
    T*     data_dev  = nullptr;
    int num_elements = 0;     //> Number of elements represented by this array.
    int index_offset = -1;    //> Offset for index calculation; scalar product of negative lower bounds and strides.
    int    stride2  = -1;    //> Stride 2 for linearizing 7-dimensional index.
    int    stride3  = -1;    //> Stride 3 for linearizing 7-dimensional index.
    int    stride4  = -1;    //> Stride 4 for linearizing 7-dimensional index.
    int    stride5  = -1;    //> Stride 5 for linearizing 7-dimensional index.
    int    stride6  = -1;    //> Stride 6 for linearizing 7-dimensional index.
    int    stride7  = -1;    //> Stride 7 for linearizing 7-dimensional index.
 
    /**
     * Wrap a host-device pointer pair and store the associated
     * array meta data.
     *
     * \param[in] data_host host data pointer.
     * \param[in] data_dev device data pointer.
     * \param[in] n1,n2,... element counts per dimension of the multi-dim. array 
     * \param[in] lb1,lb2,... lower bounds, i.e. the index of the first element per dimension of the multi-dim. array 
     */ 
    __host__ __device__ void wrap(
        T* data_host,
        T* data_dev,
        const int n1,
        const int n2,
        const int n3,
        const int n4,
        const int n5,
        const int n6,
        const int n7,
        const int lb1,
        const int lb2,
        const int lb3,
        const int lb4,
        const int lb5,
        const int lb6,
        const int lb7
      ) {
       this->data_host = data_host;
       this->data_dev  = data_dev;
       // column-major access
       this->stride2 = 1*n1;
       this->stride3 = this->stride2*n2;
       this->stride4 = this->stride3*n3;
       this->stride5 = this->stride4*n4;
       this->stride6 = this->stride5*n5;
       this->stride7 = this->stride6*n6;
       this->num_elements = this->stride7*n7;
       this->index_offset =
         -lb1*1
         -lb2*this->stride2
         -lb3*this->stride3
         -lb4*this->stride4
         -lb5*this->stride5
         -lb6*this->stride6
         -lb7*this->stride7;
    }
    
    /**
     * Linearize multi-dimensional index.
     *
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ int linearized_index (
      const int i1,
      const int i2,
      const int i3,
      const int i4,
      const int i5,
      const int i6,
      const int i7
    ) const {
      return this->index_offset
          + i1*1
          + i2*this->stride2
          + i3*this->stride3
          + i4*this->stride4
          + i5*this->stride5
          + i6*this->stride6
          + i7*this->stride7;
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ T& operator() (
      const int i1,
      const int i2,
      const int i3,
      const int i4,
      const int i5,
      const int i6,
      const int i7
    ) {
      const int index = linearized_index(
        i1,i2,i3,i4,i5,i6,i7
      );
      #ifdef __HIP_DEVICE_COMPILE__
      return this->data_dev[index];
      #else
      assert(i1 >= lbound(1));
      assert(i1 <= ubound(1));
      assert(i2 >= lbound(2));
      assert(i2 <= ubound(2));
      assert(i3 >= lbound(3));
      assert(i3 <= ubound(3));
      assert(i4 >= lbound(4));
      assert(i4 <= ubound(4));
      assert(i5 >= lbound(5));
      assert(i5 <= ubound(5));
      assert(i6 >= lbound(6));
      assert(i6 <= ubound(6));
      assert(i7 >= lbound(7));
      assert(i7 <= ubound(7));
      return this->data_host[index];
      #endif
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ const T& operator() (
      const int i1,
      const int i2,
      const int i3,
      const int i4,
      const int i5,
      const int i6,
      const int i7
    ) const {
      const int index = linearized_index(
        i1,i2,i3,i4,i5,i6,i7
      );
      #ifdef __HIP_DEVICE_COMPILE__
      return this->data_dev[index];
      #else
      assert(i1 >= lbound(1));
      assert(i1 <= ubound(1));
      assert(i2 >= lbound(2));
      assert(i2 <= ubound(2));
      assert(i3 >= lbound(3));
      assert(i3 <= ubound(3));
      assert(i4 >= lbound(4));
      assert(i4 <= ubound(4));
      assert(i5 >= lbound(5));
      assert(i5 <= ubound(5));
      assert(i6 >= lbound(6));
      assert(i6 <= ubound(6));
      assert(i7 >= lbound(7));
      assert(i7 <= ubound(7));
      return this->data_host[index];
      #endif
    }
    
    /**
     * \return Size of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,7
     */
    __host__ __device__ __forceinline__ int size(int dim) const {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(dim >= 1);
      assert(dim <= 7);
      #endif
      switch(dim) {
       case 7:
	        return this->num_elements / this->stride7;
       case 6:
	        return this->stride7 / this->stride6;
       case 5:
	        return this->stride6 / this->stride5;
       case 4:
	        return this->stride5 / this->stride4;
       case 3:
	        return this->stride4 / this->stride3;
       case 2:
	        return this->stride3 / this->stride2;
       case 1:
	        return this->stride2 / 1;
       default:
          #ifndef __HIP_DEVICE_COMPILE__
          std::cerr << "‘dim’ argument of ‘gpufort::array7::size’ is not a valid dimension index ('dim': "<<dim<<", max dimension: 7" << std::endl;
          std::terminate();
          #else
          return -1;
          #endif
      }
    }
    
    /**
     * \return Lower bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,7
     */
    __host__ __device__ __forceinline__ int lbound(int dim) const {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(dim >= 1);
      assert(dim <= 7);
      #endif
      int offset_remainder = this->index_offset;
      int lb = -offset_remainder / this->stride7;
      if ( dim == 7 ) return lb;
      offset_remainder = offset_remainder + lb*this->stride7;
      lb = -offset_remainder / this->stride6;
      if ( dim == 6 ) return lb;
      offset_remainder = offset_remainder + lb*this->stride6;
      lb = -offset_remainder / this->stride5;
      if ( dim == 5 ) return lb;
      offset_remainder = offset_remainder + lb*this->stride5;
      lb = -offset_remainder / this->stride4;
      if ( dim == 4 ) return lb;
      offset_remainder = offset_remainder + lb*this->stride4;
      lb = -offset_remainder / this->stride3;
      if ( dim == 3 ) return lb;
      offset_remainder = offset_remainder + lb*this->stride3;
      lb = -offset_remainder / this->stride2;
      if ( dim == 2 ) return lb;
      offset_remainder = offset_remainder + lb*this->stride2;
      lb = -offset_remainder / 1;
      #ifndef __HIP_DEVICE_COMPILE__
      offset_remainder = offset_remainder + lb*1;
      assert(offset_remainder == 0);
      #endif
      return lb;
    }
    
    /**
     * \return Upper bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,7
     */
    __host__ __device__ __forceinline__ int ubound(int dim) const {
      return this->lbound(dim) + this->size(dim) - 1;
    }
  };

  template<typename T>
  struct array7{
    array_descr7<T> data;
    AllocMode              alloc_mode = AllocMode::WrapHostAllocDevice; //> Data allocation strategy. Default: 
                                                                        //> wrap the host and allocate device data
    SyncMode               sync_mode  = SyncMode::None;                 //> How data should be synchronized
                                                                        //> during the initialization and destruction of this GPUFORT array.
    int num_refs             = 0;  //> Number of references.
    size_t bytes_per_element = -1; //> Bytes per element; stored to make num_data_bytes routine independent of T 

    array7() {
      // do nothing
    }
    
    ~array7() {
      // do nothing
    }

    /**
     * Initialize.
     * \param[in] data_host host data pointer (may be nullptr; see the note).
     * \param[in] pinned the host array is pinned or should be pinned.
     * \param[in] copyout_at_destruction copy data out from the
     *            device to the host when this instance gets destroyed.
     * \note Allocates a new host_data pointer if the
     * passed pointer is a nullptr. If the pinned argument is true,
     * the new hostpotr is allocated via hipHostMalloc.
     * Otherwise, it is allocated via classic malloc.
     */
    __host__ hipError_t init_async(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int n1,int n2,int n3,int n4,int n5,int n6,int n7,
        int lb1,int lb2,int lb3,int lb4,int lb5,int lb6,int lb7,
        hipStream_t stream,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      this->data.wrap(
          data_host,
          data_dev,
          n1,n2,n3,n4,n5,n6,n7,
          lb1,lb2,lb3,lb4,lb5,lb6,lb7
      );
      this->bytes_per_element = bytes_per_element;
      this->num_refs          = 1;
      this->alloc_mode        = alloc_mode;
      this->sync_mode         = sync_mode;
  
      // host array allocation
      hipError_t ierr = hipSuccess;
      switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
#endif
        case AllocMode::AllocPinnedHostWrapDevice:
        case AllocMode::AllocPinnedHostAllocDevice:
          ierr = hipHostMalloc((void**) &this->data.data_host,this->num_data_bytes(),0);
	  break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
          this->data.data_host = (T*) malloc(this->num_data_bytes());
	  break;
#endif
	case AllocMode::WrapHostWrapDevice:
        case AllocMode::WrapHostAllocDevice:
          // do nothing as already set by data.wrap call
	  break;
	default:
          std::cerr << "ERROR: gpufort::array7::init_async(...): Unexpected value for 'this->alloc_mode': " 
                    << static_cast<int>(this->alloc_mode) << std::endl; 
          std::terminate();
          break;
      }
      // device array allocation
      if ( ierr == hipSuccess ) {
        switch (this->alloc_mode) {
          case AllocMode::AllocHostWrapDevice:
          case AllocMode::AllocPinnedHostWrapDevice:
          case AllocMode::WrapHostWrapDevice:
            // do nothing as already set by data.wrap call
            break;
          case AllocMode::WrapHostAllocDevice:
          case AllocMode::AllocHostAllocDevice:
          case AllocMode::AllocPinnedHostAllocDevice:
            ierr = hipMalloc((void**) &this->data.data_dev, this->num_data_bytes());
            break;
          default:
            break;
        }
      }
      if ( ierr == hipSuccess ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::None:
          case SyncMode::Copyout:
          case SyncMode::InvertedCopyout:
            // do nothing
            break;
          case SyncMode::Copy:
          case SyncMode::Copyin:
            ierr = this->copy_to_device_async(stream);
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyin:
            ierr = this->copy_to_host_async(stream);
            break;
          default:
            std::cerr << "ERROR: gpufort::array7::init_async(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
      }
      return ierr;
    }
   
    __host__ hipError_t init_async(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int* sizes,
        int* lower_bounds,
        hipStream_t stream,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      return this->init_async(
        bytes_per_element,
        data_host,
        data_dev,
        sizes[0],sizes[1],sizes[2],sizes[3],sizes[4],sizes[5],sizes[6],
        lower_bounds[0],lower_bounds[1],lower_bounds[2],lower_bounds[3],lower_bounds[4],lower_bounds[5],lower_bounds[6],
        stream,alloc_mode,sync_mode);
    }
    
    /**
     * Destroy associated host and device data
     * if this is not a wrapper. Depending
     * on the pinned attribue, hipHostFree 
     * is used instead of free to deallocate
     * the host data.
     */
    __host__ hipError_t destroy_async(hipStream_t stream) {
      hipError_t ierr = hipSuccess;
      if ( bytes_per_element > 0 ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::Copy:
          case SyncMode::Copyout:
            ierr = this->copy_to_host_async(stream);
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyout:
            ierr = this->copy_to_device_async(stream);
            break;
          case SyncMode::None:
          case SyncMode::Copyin:
          case SyncMode::InvertedCopyin:
            // do nothing
            break;
          default:
            std::cerr << "ERROR: gpufort::array7::destroy_async(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
        // host array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
#endif
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipHostFree(this->data.data_host);
              break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
              free(this->data.data_host); 
              break;
#endif
            case AllocMode::WrapHostWrapDevice:
            case AllocMode::WrapHostAllocDevice:
              break;
            default:
              std::cerr << "ERROR: gpufort::array7::destroy_async(...): Unexpected value for 'alloc_mode': " 
                        << static_cast<int>(alloc_mode) << std::endl; 
              std::terminate();
              break;
          }
        }
        // device array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::WrapHostWrapDevice:
              // do nothing as already set by data.wrap call
              break;
            case AllocMode::WrapHostAllocDevice:
            case AllocMode::AllocHostAllocDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipFree(this->data.data_dev);
              break;
            default:
              break;
          }
        }
        if ( ierr == hipSuccess ) {
          this->data.data_host    = nullptr;
          this->data.data_dev     = nullptr;
          this->bytes_per_element = -1; 
        } 
      }
      return ierr;
    }
    
    /**
     * Copy data to a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the destinaton buffer is interpreted as host or device variable
     * and the source is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_to_buffer_async(
       void* buffer, hipMemcpyKind memcpy_kind,
       hipStream_t stream
    ) {
      T* src = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyDeviceToHost: 
        case hipMemcpyDeviceToDevice:
          src = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyHostToDevice:
          src = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array7::copy_to_buffer_async(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(src!=nullptr);
      #endif
      return hipMemcpyAsync(
        buffer, 
        (void*) src,
        this->num_data_bytes(), 
        memcpy_kind, stream);
    }
    
    /**
     * Copy data from a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the source buffer is interpreted as host or device variable
     * and the destination is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_from_buffer_async(
       void* buffer, hipMemcpyKind memcpy_kind,
       hipStream_t stream
    ) {
      T* dest = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyHostToDevice: 
        case hipMemcpyDeviceToDevice:
          dest = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyDeviceToHost:
          dest = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array7::copy_from_buffer_async(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(dest!=nullptr);
      #endif
      return hipMemcpyAsync(
        (void*) dest,
        buffer, 
        this->num_data_bytes(), 
        memcpy_kind, stream);
    }
    
    /**
     * Copy host data to the device.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_host_async(hipStream_t stream) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpyAsync(
        (void*) this->data.data_host, 
        (void*) this->data.data_dev,
        this->num_data_bytes(), 
        hipMemcpyDeviceToHost, stream);
    }
    
    /**
     * Copy device data to the host.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_device_async(hipStream_t stream) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpyAsync(
        (void*) this->data.data_dev, 
        (void*) this->data.data_host,
        this->num_data_bytes(),
        hipMemcpyHostToDevice, stream);
    }
    
    /** 
     * Copies the struct to the device into an pre-allocated
     * device memory block but does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[in] device_struct device memory address to copy to
     */
    __host__ hipError_t copy_self_to_device_async(
        gpufort::array7<T>* device_struct,
        hipStream_t stream
    ) {
      const size_t size = sizeof(array7<char>); // sizeof(T*) = sizeof(char*)
      return hipMemcpyAsync(
          (void*) device_struct, 
          (void*) this,
          size,
          hipMemcpyHostToDevice, stream);
    }

    /** 
     * Copies the struct to the device but
     * does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[inout] device_copy pointer to device copy pointer 
     */
    __host__ hipError_t create_device_copy_async(
        gpufort::array7<T>** device_copy,
        hipStream_t stream
    ) {
      const size_t size = sizeof(array7<char>); // sizeof(T*) = sizeof(char*)
      hipError_t ierr   = hipMalloc((void**)device_copy,size);      
      if ( ierr == hipSuccess ) {
        return this->copy_self_to_device_async(*device_copy,stream);
      } else {
        return ierr;
      }
    }
    /**
     * Initialize.
     * \param[in] data_host host data pointer (may be nullptr; see the note).
     * \param[in] pinned the host array is pinned or should be pinned.
     * \param[in] copyout_at_destruction copy data out from the
     *            device to the host when this instance gets destroyed.
     * \note Allocates a new host_data pointer if the
     * passed pointer is a nullptr. If the pinned argument is true,
     * the new hostpotr is allocated via hipHostMalloc.
     * Otherwise, it is allocated via classic malloc.
     */
    __host__ hipError_t init(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int n1,int n2,int n3,int n4,int n5,int n6,int n7,
        int lb1,int lb2,int lb3,int lb4,int lb5,int lb6,int lb7,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      this->data.wrap(
          data_host,
          data_dev,
          n1,n2,n3,n4,n5,n6,n7,
          lb1,lb2,lb3,lb4,lb5,lb6,lb7
      );
      this->bytes_per_element = bytes_per_element;
      this->num_refs          = 1;
      this->alloc_mode        = alloc_mode;
      this->sync_mode         = sync_mode;
  
      // host array allocation
      hipError_t ierr = hipSuccess;
      switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
#endif
        case AllocMode::AllocPinnedHostWrapDevice:
        case AllocMode::AllocPinnedHostAllocDevice:
          ierr = hipHostMalloc((void**) &this->data.data_host,this->num_data_bytes(),0);
	  break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
          this->data.data_host = (T*) malloc(this->num_data_bytes());
	  break;
#endif
	case AllocMode::WrapHostWrapDevice:
        case AllocMode::WrapHostAllocDevice:
          // do nothing as already set by data.wrap call
	  break;
	default:
          std::cerr << "ERROR: gpufort::array7::init(...): Unexpected value for 'this->alloc_mode': " 
                    << static_cast<int>(this->alloc_mode) << std::endl; 
          std::terminate();
          break;
      }
      // device array allocation
      if ( ierr == hipSuccess ) {
        switch (this->alloc_mode) {
          case AllocMode::AllocHostWrapDevice:
          case AllocMode::AllocPinnedHostWrapDevice:
          case AllocMode::WrapHostWrapDevice:
            // do nothing as already set by data.wrap call
            break;
          case AllocMode::WrapHostAllocDevice:
          case AllocMode::AllocHostAllocDevice:
          case AllocMode::AllocPinnedHostAllocDevice:
            ierr = hipMalloc((void**) &this->data.data_dev, this->num_data_bytes());
            break;
          default:
            break;
        }
      }
      if ( ierr == hipSuccess ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::None:
          case SyncMode::Copyout:
          case SyncMode::InvertedCopyout:
            // do nothing
            break;
          case SyncMode::Copy:
          case SyncMode::Copyin:
            ierr = this->copy_to_device();
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyin:
            ierr = this->copy_to_host();
            break;
          default:
            std::cerr << "ERROR: gpufort::array7::init(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
      }
      return ierr;
    }
   
    __host__ hipError_t init(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int* sizes,
        int* lower_bounds,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      return this->init(
        bytes_per_element,
        data_host,
        data_dev,
        sizes[0],sizes[1],sizes[2],sizes[3],sizes[4],sizes[5],sizes[6],
        lower_bounds[0],lower_bounds[1],lower_bounds[2],lower_bounds[3],lower_bounds[4],lower_bounds[5],lower_bounds[6],alloc_mode,sync_mode);
    }
    
    /**
     * Destroy associated host and device data
     * if this is not a wrapper. Depending
     * on the pinned attribue, hipHostFree 
     * is used instead of free to deallocate
     * the host data.
     */
    __host__ hipError_t destroy() {
      hipError_t ierr = hipSuccess;
      if ( bytes_per_element > 0 ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::Copy:
          case SyncMode::Copyout:
            ierr = this->copy_to_host();
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyout:
            ierr = this->copy_to_device();
            break;
          case SyncMode::None:
          case SyncMode::Copyin:
          case SyncMode::InvertedCopyin:
            // do nothing
            break;
          default:
            std::cerr << "ERROR: gpufort::array7::destroy(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
        // host array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
#endif
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipHostFree(this->data.data_host);
              break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
              free(this->data.data_host); 
              break;
#endif
            case AllocMode::WrapHostWrapDevice:
            case AllocMode::WrapHostAllocDevice:
              break;
            default:
              std::cerr << "ERROR: gpufort::array7::destroy(...): Unexpected value for 'alloc_mode': " 
                        << static_cast<int>(alloc_mode) << std::endl; 
              std::terminate();
              break;
          }
        }
        // device array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::WrapHostWrapDevice:
              // do nothing as already set by data.wrap call
              break;
            case AllocMode::WrapHostAllocDevice:
            case AllocMode::AllocHostAllocDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipFree(this->data.data_dev);
              break;
            default:
              break;
          }
        }
        if ( ierr == hipSuccess ) {
          this->data.data_host    = nullptr;
          this->data.data_dev     = nullptr;
          this->bytes_per_element = -1; 
        } 
      }
      return ierr;
    }
    
    /**
     * Copy data to a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the destinaton buffer is interpreted as host or device variable
     * and the source is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_to_buffer(
       void* buffer, hipMemcpyKind memcpy_kind
    ) {
      T* src = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyDeviceToHost: 
        case hipMemcpyDeviceToDevice:
          src = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyHostToDevice:
          src = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array7::copy_to_buffer(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(src!=nullptr);
      #endif
      return hipMemcpy(
        buffer, 
        (void*) src,
        this->num_data_bytes(), 
        memcpy_kind);
    }
    
    /**
     * Copy data from a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the source buffer is interpreted as host or device variable
     * and the destination is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_from_buffer(
       void* buffer, hipMemcpyKind memcpy_kind
    ) {
      T* dest = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyHostToDevice: 
        case hipMemcpyDeviceToDevice:
          dest = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyDeviceToHost:
          dest = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array7::copy_from_buffer(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(dest!=nullptr);
      #endif
      return hipMemcpy(
        (void*) dest,
        buffer, 
        this->num_data_bytes(), 
        memcpy_kind);
    }
    
    /**
     * Copy host data to the device.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_host() {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpy(
        (void*) this->data.data_host, 
        (void*) this->data.data_dev,
        this->num_data_bytes(), 
        hipMemcpyDeviceToHost);
    }
    
    /**
     * Copy device data to the host.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_device() {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpy(
        (void*) this->data.data_dev, 
        (void*) this->data.data_host,
        this->num_data_bytes(),
        hipMemcpyHostToDevice);
    }
    
    /** 
     * Copies the struct to the device into an pre-allocated
     * device memory block but does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[in] device_struct device memory address to copy to
     */
    __host__ hipError_t copy_self_to_device(
        gpufort::array7<T>* device_struct
    ) {
      const size_t size = sizeof(array7<char>); // sizeof(T*) = sizeof(char*)
      return hipMemcpy(
          (void*) device_struct, 
          (void*) this,
          size,
          hipMemcpyHostToDevice);
    }

    /** 
     * Copies the struct to the device but
     * does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[inout] device_copy pointer to device copy pointer 
     */
    __host__ hipError_t create_device_copy(
        gpufort::array7<T>** device_copy
    ) {
      const size_t size = sizeof(array7<char>); // sizeof(T*) = sizeof(char*)
      hipError_t ierr   = hipMalloc((void**)device_copy,size);      
      if ( ierr == hipSuccess ) {
        return this->copy_self_to_device(*device_copy);
      } else {
        return ierr;
      }
    }

    /**
     * Allocate a host buffer with
     * the same size as the data buffers
     * associated with this gpufort array.
     * @see num_data_bytes()
     * \param[inout] pointer to the buffer to allocate
     * \param[in] pinned If the memory should be pinned (default=True)
     * \param[in] flags  Flags for the host memory allocation (default=0).
     */
    __host__ hipError_t allocate_host_buffer(void** buffer,bool pinned=true,int flags=0) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
      if ( pinned ) { 
        ierr = hipHostMalloc(buffer,this->num_data_bytes(),flags);
      } else {
        *buffer = malloc(this->num_data_bytes());
      }
      return ierr;
    }

    /**
     * Deallocate a host buffer
     * created via the allocate_host_buffer routine.
     * @see num_data_bytes(), allocate_host_buffer
     * \param[inout] the buffer to deallocte
     * \param[in] pinned If the memory to deallocate is pinned (default=True)
     */
    __host__ hipError_t deallocate_host_buffer(void* buffer,bool pinned=true) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
      if ( pinned ) { 
        ierr = hipHostFree(buffer);
      } else {
        free(buffer);
      }
      return ierr;
    }
    /**
     * Allocate a device buffer with
     * the same size as the data buffers
     * associated with this gpufort array.
     * @see num_data_bytes()
     * \param[inout] pointer to the buffer to allocate
     */
    __host__ hipError_t allocate_device_buffer(void** buffer) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
 
      ierr = hipMalloc(buffer,this->num_data_bytes());
      return ierr;
    }

    /**
     * Deallocate a device buffer
     * created via the allocate_device_buffer routine.
     * @see num_data_bytes(), allocate_device_buffer
     * \param[inout] the buffer to deallocte
     */
    __host__ hipError_t deallocate_device_buffer(void* buffer) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
 
      ierr = hipFree(buffer);
      return ierr;
    }

    /** 
     * Copy metadata from another gpufort array to this gpufort array.
     * \note The allocation mode is always set to AllocMode::WrapHostWrapDevice
     *       and the sync mode is set to SyncMode::None.
     * \note No deep copy, i.e. the host and device data is not copied, just the pointers
     * to the data.
     */
    __host__ void copy(const gpufort::array7<T>& other) {
      this->init(
        other.bytes_per_element,
        other.data.data_host,
        other.data.data_dev,
        other.size(1),other.size(2),other.size(3),other.size(4),other.size(5),other.size(6),other.size(7),
        other.lbound(1),other.lbound(2),other.lbound(3),other.lbound(4),other.lbound(5),other.lbound(6),other.lbound(7),
        AllocMode::WrapHostWrapDevice,
        SyncMode::None);
    }

    /** 
     *  \return Number of bytes required to store 
     *  the array elements on the host or the device.
     */ 
    __host__ size_t num_data_bytes() {
      return this->data.num_elements * this->bytes_per_element;
    }
    
    /**
     * \return the number of elements of this array.
     */
    __host__ int num_elements() {
      return this->data.num_elements;
    }

    /**
     * Linearize multi-dimensional index.
     *
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ int linearized_index (
      const int i1,
      const int i2,
      const int i3,
      const int i4,
      const int i5,
      const int i6,
      const int i7
    ) const {
      return this->data.linearized_index(
        i1,i2,i3,i4,i5,i6,i7
      );
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ T& operator() (
      const int i1,
      const int i2,
      const int i3,
      const int i4,
      const int i5,
      const int i6,
      const int i7
    ) {
      return this->data(
        i1,i2,i3,i4,i5,i6,i7
      );
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ const T& operator() (
      const int i1,
      const int i2,
      const int i3,
      const int i4,
      const int i5,
      const int i6,
      const int i7
    ) const {
      return this->data(
        i1,i2,i3,i4,i5,i6,i7
      );
    }
    
    /**
     * \return Size of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,7
     */
    __host__ __device__ __forceinline__ int size(int dim) const {
      return this->data.size(dim);
    }
    
    /**
     * \return Lower bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,7
     */
    __host__ __device__ __forceinline__ int lbound(int dim) const {
      return this->data.lbound(dim);
    }
    
    /**
     * \return Upper bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,7
     */
    __host__ __device__ __forceinline__ int ubound(int dim) const {
      return this->data.ubound(dim);
    }
    

    /**
     * Collapse the array by fixing 1 indices.
     * \return A gpufort array of rank 6.
     * \param[in] i7,...,i7 indices to fix.
     */
    __host__ gpufort::array6<T> collapse(
        const int i7
    ) const {
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1);
      const int n2  = this->size(2);
      const int lb2 = this->lbound(2);
      const int n3  = this->size(3);
      const int lb3 = this->lbound(3);
      const int n4  = this->size(4);
      const int lb4 = this->lbound(4);
      const int n5  = this->size(5);
      const int lb5 = this->lbound(5);
      const int n6  = this->size(6);
      const int lb6 = this->lbound(6);
      const int index = linearized_index(
        lb1,lb2,lb3,lb4,lb5,lb6,
        i7);
      T* data_host_new = nullptr; 
      if (this->data.data_host != nullptr) {
        data_host_new = this->data.data_host + index;
      }
      T* data_dev_new = nullptr; 
      if (this->data.data_dev != nullptr) {
        data_dev_new = this->data.data_dev + index;
      }
      gpufort::array6<T> collapsed_array;
      hipError_t ierr = 
        collapsed_array.init(
          this->bytes_per_element,
          data_host_new,
          data_dev_new,
          n1,n2,n3,n4,n5,n6,
          lb1,lb2,lb3,lb4,lb5,lb6,
          AllocMode::WrapHostWrapDevice,
          SyncMode::None);
      return collapsed_array;
    }
    /**
     * Collapse the array by fixing 2 indices.
     * \return A gpufort array of rank 5.
     * \param[in] i6,...,i7 indices to fix.
     */
    __host__ gpufort::array5<T> collapse(
        const int i6,
        const int i7
    ) const {
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1);
      const int n2  = this->size(2);
      const int lb2 = this->lbound(2);
      const int n3  = this->size(3);
      const int lb3 = this->lbound(3);
      const int n4  = this->size(4);
      const int lb4 = this->lbound(4);
      const int n5  = this->size(5);
      const int lb5 = this->lbound(5);
      const int index = linearized_index(
        lb1,lb2,lb3,lb4,lb5,
        i6,i7);
      T* data_host_new = nullptr; 
      if (this->data.data_host != nullptr) {
        data_host_new = this->data.data_host + index;
      }
      T* data_dev_new = nullptr; 
      if (this->data.data_dev != nullptr) {
        data_dev_new = this->data.data_dev + index;
      }
      gpufort::array5<T> collapsed_array;
      hipError_t ierr = 
        collapsed_array.init(
          this->bytes_per_element,
          data_host_new,
          data_dev_new,
          n1,n2,n3,n4,n5,
          lb1,lb2,lb3,lb4,lb5,
          AllocMode::WrapHostWrapDevice,
          SyncMode::None);
      return collapsed_array;
    }
    /**
     * Collapse the array by fixing 3 indices.
     * \return A gpufort array of rank 4.
     * \param[in] i5,...,i7 indices to fix.
     */
    __host__ gpufort::array4<T> collapse(
        const int i5,
        const int i6,
        const int i7
    ) const {
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1);
      const int n2  = this->size(2);
      const int lb2 = this->lbound(2);
      const int n3  = this->size(3);
      const int lb3 = this->lbound(3);
      const int n4  = this->size(4);
      const int lb4 = this->lbound(4);
      const int index = linearized_index(
        lb1,lb2,lb3,lb4,
        i5,i6,i7);
      T* data_host_new = nullptr; 
      if (this->data.data_host != nullptr) {
        data_host_new = this->data.data_host + index;
      }
      T* data_dev_new = nullptr; 
      if (this->data.data_dev != nullptr) {
        data_dev_new = this->data.data_dev + index;
      }
      gpufort::array4<T> collapsed_array;
      hipError_t ierr = 
        collapsed_array.init(
          this->bytes_per_element,
          data_host_new,
          data_dev_new,
          n1,n2,n3,n4,
          lb1,lb2,lb3,lb4,
          AllocMode::WrapHostWrapDevice,
          SyncMode::None);
      return collapsed_array;
    }
    /**
     * Collapse the array by fixing 4 indices.
     * \return A gpufort array of rank 3.
     * \param[in] i4,...,i7 indices to fix.
     */
    __host__ gpufort::array3<T> collapse(
        const int i4,
        const int i5,
        const int i6,
        const int i7
    ) const {
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1);
      const int n2  = this->size(2);
      const int lb2 = this->lbound(2);
      const int n3  = this->size(3);
      const int lb3 = this->lbound(3);
      const int index = linearized_index(
        lb1,lb2,lb3,
        i4,i5,i6,i7);
      T* data_host_new = nullptr; 
      if (this->data.data_host != nullptr) {
        data_host_new = this->data.data_host + index;
      }
      T* data_dev_new = nullptr; 
      if (this->data.data_dev != nullptr) {
        data_dev_new = this->data.data_dev + index;
      }
      gpufort::array3<T> collapsed_array;
      hipError_t ierr = 
        collapsed_array.init(
          this->bytes_per_element,
          data_host_new,
          data_dev_new,
          n1,n2,n3,
          lb1,lb2,lb3,
          AllocMode::WrapHostWrapDevice,
          SyncMode::None);
      return collapsed_array;
    }
    /**
     * Collapse the array by fixing 5 indices.
     * \return A gpufort array of rank 2.
     * \param[in] i3,...,i7 indices to fix.
     */
    __host__ gpufort::array2<T> collapse(
        const int i3,
        const int i4,
        const int i5,
        const int i6,
        const int i7
    ) const {
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1);
      const int n2  = this->size(2);
      const int lb2 = this->lbound(2);
      const int index = linearized_index(
        lb1,lb2,
        i3,i4,i5,i6,i7);
      T* data_host_new = nullptr; 
      if (this->data.data_host != nullptr) {
        data_host_new = this->data.data_host + index;
      }
      T* data_dev_new = nullptr; 
      if (this->data.data_dev != nullptr) {
        data_dev_new = this->data.data_dev + index;
      }
      gpufort::array2<T> collapsed_array;
      hipError_t ierr = 
        collapsed_array.init(
          this->bytes_per_element,
          data_host_new,
          data_dev_new,
          n1,n2,
          lb1,lb2,
          AllocMode::WrapHostWrapDevice,
          SyncMode::None);
      return collapsed_array;
    }
    /**
     * Collapse the array by fixing 6 indices.
     * \return A gpufort array of rank 1.
     * \param[in] i2,...,i7 indices to fix.
     */
    __host__ gpufort::array1<T> collapse(
        const int i2,
        const int i3,
        const int i4,
        const int i5,
        const int i6,
        const int i7
    ) const {
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1);
      const int index = linearized_index(
        lb1,
        i2,i3,i4,i5,i6,i7);
      T* data_host_new = nullptr; 
      if (this->data.data_host != nullptr) {
        data_host_new = this->data.data_host + index;
      }
      T* data_dev_new = nullptr; 
      if (this->data.data_dev != nullptr) {
        data_dev_new = this->data.data_dev + index;
      }
      gpufort::array1<T> collapsed_array;
      hipError_t ierr = 
        collapsed_array.init(
          this->bytes_per_element,
          data_host_new,
          data_dev_new,
          n1,
          lb1,
          AllocMode::WrapHostWrapDevice,
          SyncMode::None);
      return collapsed_array;
    }

    /**
     * Write host data values and their corresponding index
     * to an output stream.
     * \param[inout] out        output stream
     * \param[in]    prefix     prefix to put before each output line
     * \param[in]    print_prec precision to use when printing floating point numbers 
     *               [default=GPUFORT_ARRAY_PRINT_PREC[default=6]]
     * \note This method only makes sense for numeric data types and
     *       may result in compilation, runtime errors or undefined output if the
     *       underlying data is not numeric.
     */
    __host__ void print_host_data(
        std::ostream& out, const char* prefix, PrintMode print_mode,
        const int print_prec=GPUFORT_ARRAY_PRINT_PREC) const {
      const bool print_norms = 
	print_mode==PrintMode::PrintNorms ||
        print_mode==PrintMode::PrintValuesAndNorms ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      const bool print_values = 
        print_mode==PrintMode::PrintValues ||
        print_mode==PrintMode::PrintValuesAndNorms;
      const bool print_values_c_style = 
        print_mode==PrintMode::PrintCValues ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      if ( !print_norms && !print_values && !print_values_c_style ) {
        std::cerr << "ERROR: gpufort::array7::print_host_data(...): Unexpected value for 'print_mode': " 
                  << static_cast<int>(print_mode) << std::endl; 
        std::terminate();
      }
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1); 
      const int n2  = this->size(2);
      const int lb2 = this->lbound(2); 
      const int n3  = this->size(3);
      const int lb3 = this->lbound(3); 
      const int n4  = this->size(4);
      const int lb4 = this->lbound(4); 
      const int n5  = this->size(5);
      const int lb5 = this->lbound(5); 
      const int n6  = this->size(6);
      const int lb6 = this->lbound(6); 
      const int n7  = this->size(7);
      const int lb7 = this->lbound(7); 
      const int n = this->data.num_elements; 
      T* A_h   = this->data.data_host;
      T min = +std::numeric_limits<T>::max();
      T max = -std::numeric_limits<T>::max();
      T sum = 0;
      T l1  = 0;
      T l2  = 0;
      out << prefix << ":\n";
    for ( int i7 = 0; i7 < n7; i7++ ) {
    for ( int i6 = 0; i6 < n6; i6++ ) {
    for ( int i5 = 0; i5 < n5; i5++ ) {
    for ( int i4 = 0; i4 < n4; i4++ ) {
    for ( int i3 = 0; i3 < n3; i3++ ) {
    for ( int i2 = 0; i2 < n2; i2++ ) {
    for ( int i1 = 0; i1 < n1; i1++ ) {
      const int idx = this->linearized_index(i1,i2,i3,i4,i5,i6,i7);
      T value = A_h[idx];
      if ( print_norms ) {
        min  = std::min(value,min);
        max  = std::max(value,max);
        sum += value;
        l1  += std::abs(value);
        l2  += value*value;
      }
      if ( print_values ) {
          out << prefix << "(" << (lb1+i1) << "," << (lb2+i2) << "," << (lb3+i3) << "," << (lb4+i4) << "," << (lb5+i5) << "," << (lb6+i6) << "," << (lb7+i7) <<  ") = " << std::setprecision(print_prec) << value << "\n";
      } else if ( print_values_c_style ) {
          out << prefix << "[" << idx << "] = " << std::setprecision(print_prec) << value << "\n";
      }
    }}}}}}} // for loops
    if ( print_norms ) {
      out << prefix << ":min=" << std::setprecision(print_prec) << min << "\n";
      out << prefix << ":max=" << std::setprecision(print_prec) << max << "\n";
      out << prefix << ":sum=" << std::setprecision(print_prec) << sum << "\n";
      out << prefix << ":l1="  << std::setprecision(print_prec) << l1  << "\n";
      out << prefix << ":l2="  << std::setprecision(print_prec) << std::sqrt(l2) << "\n";
      out << prefix << "num_elements=" << std::setprecision(print_prec) << n << "\n";
    }
  }
    /**
     * Write device data values and their corresponding index
     * to an output stream.
     * \param[inout] out        output stream
     * \param[in]    prefix     prefix to put before each output line
     * \param[in]    print_prec precision to use when printing floating point numbers 
     *               [default=GPUFORT_ARRAY_PRINT_PREC[default=6]]
     * \note This method only makes sense for numeric data types and
     *       may result in compilation, runtime errors or undefined output if the
     *       underlying data is not numeric.
     */
    __host__ void print_device_data(
        std::ostream& out, const char* prefix, PrintMode print_mode,
        const int print_prec=GPUFORT_ARRAY_PRINT_PREC) const {
      const bool print_norms = 
	print_mode==PrintMode::PrintNorms ||
        print_mode==PrintMode::PrintValuesAndNorms ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      const bool print_values = 
        print_mode==PrintMode::PrintValues ||
        print_mode==PrintMode::PrintValuesAndNorms;
      const bool print_values_c_style = 
        print_mode==PrintMode::PrintCValues ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      if ( !print_norms && !print_values && !print_values_c_style ) {
        std::cerr << "ERROR: gpufort::array7::print_device_data(...): Unexpected value for 'print_mode': " 
                  << static_cast<int>(print_mode) << std::endl; 
        std::terminate();
      }
      const int n1  = this->size(1);
      const int lb1 = this->lbound(1); 
      const int n2  = this->size(2);
      const int lb2 = this->lbound(2); 
      const int n3  = this->size(3);
      const int lb3 = this->lbound(3); 
      const int n4  = this->size(4);
      const int lb4 = this->lbound(4); 
      const int n5  = this->size(5);
      const int lb5 = this->lbound(5); 
      const int n6  = this->size(6);
      const int lb6 = this->lbound(6); 
      const int n7  = this->size(7);
      const int lb7 = this->lbound(7); 
      const int n = this->data.num_elements; 
      std::vector<T> host_array(n);
      T* A_h = host_array.data();
      T* A   = this->data.data_dev;
      HIP_CHECK(hipMemcpy(A_h, A, n*sizeof(T), hipMemcpyDeviceToHost));
      T min = +std::numeric_limits<T>::max();
      T max = -std::numeric_limits<T>::max();
      T sum = 0;
      T l1  = 0;
      T l2  = 0;
      out << prefix << ":\n";
    for ( int i7 = 0; i7 < n7; i7++ ) {
    for ( int i6 = 0; i6 < n6; i6++ ) {
    for ( int i5 = 0; i5 < n5; i5++ ) {
    for ( int i4 = 0; i4 < n4; i4++ ) {
    for ( int i3 = 0; i3 < n3; i3++ ) {
    for ( int i2 = 0; i2 < n2; i2++ ) {
    for ( int i1 = 0; i1 < n1; i1++ ) {
      const int idx = this->linearized_index(i1,i2,i3,i4,i5,i6,i7);
      T value = A_h[idx];
      if ( print_norms ) {
        min  = std::min(value,min);
        max  = std::max(value,max);
        sum += value;
        l1  += std::abs(value);
        l2  += value*value;
      }
      if ( print_values ) {
          out << prefix << "(" << (lb1+i1) << "," << (lb2+i2) << "," << (lb3+i3) << "," << (lb4+i4) << "," << (lb5+i5) << "," << (lb6+i6) << "," << (lb7+i7) <<  ") = " << std::setprecision(print_prec) << value << "\n";
      } else if ( print_values_c_style ) {
          out << prefix << "[" << idx << "] = " << std::setprecision(print_prec) << value << "\n";
      }
    }}}}}}} // for loops
    if ( print_norms ) {
      out << prefix << ":min=" << std::setprecision(print_prec) << min << "\n";
      out << prefix << ":max=" << std::setprecision(print_prec) << max << "\n";
      out << prefix << ":sum=" << std::setprecision(print_prec) << sum << "\n";
      out << prefix << ":l1="  << std::setprecision(print_prec) << l1  << "\n";
      out << prefix << ":l2="  << std::setprecision(print_prec) << std::sqrt(l2) << "\n";
      out << prefix << "num_elements=" << std::setprecision(print_prec) << n << "\n";
    }
  }

  };

}
#endif // _GPUFORT_ARRAYS_H_
