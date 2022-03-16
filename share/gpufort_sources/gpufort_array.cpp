// This file was generated from a template via gpufort --gpufort-create-headers
// C bindings
#include "gpufort_array.h"

extern "C" {
  __host__ hipError_t gpufort_array1_init_async (
      gpufort::array1<char>* array,
      int bytes_per_element,
      char* data_host,
      char* data_dev,
      int* sizes,
      int* lower_bounds,
      hipStream_t stream,
      gpufort::AllocMode alloc_mode,
      gpufort::SyncMode sync_mode
  ) {
    return array->init_async(
      bytes_per_element,
      data_host, data_dev,
      sizes, lower_bounds,
      stream,alloc_mode, sync_mode
    );
  } 
  
  __host__ hipError_t gpufort_array1_destroy_async(
      gpufort::array1<char>* array,
      hipStream_t stream
  ) {
    return array->destroy_async(stream);
  } 
  
  __host__ hipError_t gpufort_array1_copy_to_buffer_async (
      gpufort::array1<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind,
      hipStream_t stream
  ) {
    return array->copy_to_buffer_async(buffer,memcpy_kind,stream);
  } 
  
  __host__ hipError_t gpufort_array1_copy_from_buffer_async (
      gpufort::array1<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind,
      hipStream_t stream
  ) {
    return array->copy_from_buffer_async(buffer,memcpy_kind,stream);
  } 
  
  __host__ hipError_t gpufort_array1_copy_to_host_async (
      gpufort::array1<char>* array,
      hipStream_t stream
  ) {
    return array->copy_to_host_async(stream);
  } 
  
  __host__ hipError_t gpufort_array1_copy_to_device_async (
      gpufort::array1<char>* array,
      hipStream_t stream
  ) {
    return array->copy_to_device_async(stream);
  } 
  
  __host__ hipError_t gpufort_array1_dec_num_refs_async(
      gpufort::array1<char>* array,
      bool destroy_if_zero_refs,
      hipStream_t stream
  ) {
    array->num_refs -= 1;
    if ( destroy_if_zero_refs && array->num_refs == 0 ) {
      return array->destroy_async(stream);
    } {
      return hipSuccess;
    }
  } 
  __host__ hipError_t gpufort_array1_init (
      gpufort::array1<char>* array,
      int bytes_per_element,
      char* data_host,
      char* data_dev,
      int* sizes,
      int* lower_bounds,
      gpufort::AllocMode alloc_mode,
      gpufort::SyncMode sync_mode
  ) {
    return array->init(
      bytes_per_element,
      data_host, data_dev,
      sizes, lower_bounds,
      alloc_mode, sync_mode
    );
  } 
  
  __host__ hipError_t gpufort_array1_destroy(
      gpufort::array1<char>* array
  ) {
    return array->destroy();
  } 
  
  __host__ hipError_t gpufort_array1_copy_to_buffer (
      gpufort::array1<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind
  ) {
    return array->copy_to_buffer(buffer,memcpy_kind);
  } 
  
  __host__ hipError_t gpufort_array1_copy_from_buffer (
      gpufort::array1<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind
  ) {
    return array->copy_from_buffer(buffer,memcpy_kind);
  } 
  
  __host__ hipError_t gpufort_array1_copy_to_host (
      gpufort::array1<char>* array
  ) {
    return array->copy_to_host();
  } 
  
  __host__ hipError_t gpufort_array1_copy_to_device (
      gpufort::array1<char>* array
  ) {
    return array->copy_to_device();
  } 
  
  __host__ hipError_t gpufort_array1_dec_num_refs(
      gpufort::array1<char>* array,
      bool destroy_if_zero_refs
  ) {
    array->num_refs -= 1;
    if ( destroy_if_zero_refs && array->num_refs == 0 ) {
      return array->destroy();
    } {
      return hipSuccess;
    }
  } 

  /**
   * Allocate a host buffer with
   * the same size as the data buffers
   * associated with this gpufort array.
   * \see num_data_bytes()
   * \param[inout] pointer to the buffer to allocate
   * \param[in] pinned If the memory should be pinned (default=true)
   * \param[in] flags  Flags for the host memory allocation (default=0).
   */
  __host__ hipError_t gpufort_array1_allocate_host_buffer(
    gpufort::array1<char>* array,
    void** buffer,bool pinned=true,int flags=0
  ) {
    return array->allocate_host_buffer(buffer,pinned,flags);
  }

  /**
   * Deallocate a host buffer
   * created via the allocate_host_buffer routine.
   * \see num_data_bytes(), allocate_host_buffer
   * \param[inout] the buffer to deallocte
     * \param[in] pinned If the memory to deallocate is pinned [default=true]
   */
  __host__ hipError_t gpufort_array1_deallocate_host_buffer(
      gpufort::array1<char>* array,
      void* buffer,bool pinned=true) {
    return array->deallocate_host_buffer(buffer,pinned);
  }
  /**
   * Allocate a device buffer with
   * the same size as the data buffers
   * associated with this gpufort array.
   * \see num_data_bytes()
   * \param[inout] pointer to the buffer to allocate
   */
  __host__ hipError_t gpufort_array1_allocate_device_buffer(
    gpufort::array1<char>* array,
    void** buffer
  ) {
    return array->allocate_device_buffer(buffer);
  }

  /**
   * Deallocate a device buffer
   * created via the allocate_device_buffer routine.
   * \see num_data_bytes(), allocate_device_buffer
   * \param[inout] the buffer to deallocte
   */
  __host__ hipError_t gpufort_array1_deallocate_device_buffer(
      gpufort::array1<char>* array,
      void* buffer) {
    return array->deallocate_device_buffer(buffer);
  }
   
  /**
   * \return size of the array in dimension 'dim'.
   * \param[in] dim selected dimension: 1,...,1
   */
  __host__  int gpufort_array1_size(
      gpufort::array1<char>* array,
      int dim) {
    return array->data.size(dim);
  }
  
  /**
   * \return lower bound (inclusive) of the array in dimension 'dim'.
   * \param[in] dim selected dimension: 1,...,1
   */
  __host__  int gpufort_array1_lbound(
      gpufort::array1<char>* array,
      int dim) {
    return array->data.lbound(dim);
  }
  
  /**
   * \return upper bound (inclusive) of the array in dimension 'dim'.
   * \param[in] dim selected dimension: 1,...,1
   */
  __host__  int gpufort_array1_ubound(
      gpufort::array1<char>* array,
      int dim) {
    return array->data.ubound(dim);
  }

  
  __host__ void gpufort_array1_inc_num_refs(
      gpufort::array1<char>* array
  ) {
    array->num_refs += 1;
  } 
  __host__ hipError_t gpufort_array2_init_async (
      gpufort::array2<char>* array,
      int bytes_per_element,
      char* data_host,
      char* data_dev,
      int* sizes,
      int* lower_bounds,
      hipStream_t stream,
      gpufort::AllocMode alloc_mode,
      gpufort::SyncMode sync_mode
  ) {
    return array->init_async(
      bytes_per_element,
      data_host, data_dev,
      sizes, lower_bounds,
      stream,alloc_mode, sync_mode
    );
  } 
  
  __host__ hipError_t gpufort_array2_destroy_async(
      gpufort::array2<char>* array,
      hipStream_t stream
  ) {
    return array->destroy_async(stream);
  } 
  
  __host__ hipError_t gpufort_array2_copy_to_buffer_async (
      gpufort::array2<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind,
      hipStream_t stream
  ) {
    return array->copy_to_buffer_async(buffer,memcpy_kind,stream);
  } 
  
  __host__ hipError_t gpufort_array2_copy_from_buffer_async (
      gpufort::array2<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind,
      hipStream_t stream
  ) {
    return array->copy_from_buffer_async(buffer,memcpy_kind,stream);
  } 
  
  __host__ hipError_t gpufort_array2_copy_to_host_async (
      gpufort::array2<char>* array,
      hipStream_t stream
  ) {
    return array->copy_to_host_async(stream);
  } 
  
  __host__ hipError_t gpufort_array2_copy_to_device_async (
      gpufort::array2<char>* array,
      hipStream_t stream
  ) {
    return array->copy_to_device_async(stream);
  } 
  
  __host__ hipError_t gpufort_array2_dec_num_refs_async(
      gpufort::array2<char>* array,
      bool destroy_if_zero_refs,
      hipStream_t stream
  ) {
    array->num_refs -= 1;
    if ( destroy_if_zero_refs && array->num_refs == 0 ) {
      return array->destroy_async(stream);
    } {
      return hipSuccess;
    }
  } 
  __host__ hipError_t gpufort_array2_init (
      gpufort::array2<char>* array,
      int bytes_per_element,
      char* data_host,
      char* data_dev,
      int* sizes,
      int* lower_bounds,
      gpufort::AllocMode alloc_mode,
      gpufort::SyncMode sync_mode
  ) {
    return array->init(
      bytes_per_element,
      data_host, data_dev,
      sizes, lower_bounds,
      alloc_mode, sync_mode
    );
  } 
  
  __host__ hipError_t gpufort_array2_destroy(
      gpufort::array2<char>* array
  ) {
    return array->destroy();
  } 
  
  __host__ hipError_t gpufort_array2_copy_to_buffer (
      gpufort::array2<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind
  ) {
    return array->copy_to_buffer(buffer,memcpy_kind);
  } 
  
  __host__ hipError_t gpufort_array2_copy_from_buffer (
      gpufort::array2<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind
  ) {
    return array->copy_from_buffer(buffer,memcpy_kind);
  } 
  
  __host__ hipError_t gpufort_array2_copy_to_host (
      gpufort::array2<char>* array
  ) {
    return array->copy_to_host();
  } 
  
  __host__ hipError_t gpufort_array2_copy_to_device (
      gpufort::array2<char>* array
  ) {
    return array->copy_to_device();
  } 
  
  __host__ hipError_t gpufort_array2_dec_num_refs(
      gpufort::array2<char>* array,
      bool destroy_if_zero_refs
  ) {
    array->num_refs -= 1;
    if ( destroy_if_zero_refs && array->num_refs == 0 ) {
      return array->destroy();
    } {
      return hipSuccess;
    }
  } 

  /**
   * Allocate a host buffer with
   * the same size as the data buffers
   * associated with this gpufort array.
   * \see num_data_bytes()
   * \param[inout] pointer to the buffer to allocate
   * \param[in] pinned If the memory should be pinned (default=true)
   * \param[in] flags  Flags for the host memory allocation (default=0).
   */
  __host__ hipError_t gpufort_array2_allocate_host_buffer(
    gpufort::array2<char>* array,
    void** buffer,bool pinned=true,int flags=0
  ) {
    return array->allocate_host_buffer(buffer,pinned,flags);
  }

  /**
   * Deallocate a host buffer
   * created via the allocate_host_buffer routine.
   * \see num_data_bytes(), allocate_host_buffer
   * \param[inout] the buffer to deallocte
     * \param[in] pinned If the memory to deallocate is pinned [default=true]
   */
  __host__ hipError_t gpufort_array2_deallocate_host_buffer(
      gpufort::array2<char>* array,
      void* buffer,bool pinned=true) {
    return array->deallocate_host_buffer(buffer,pinned);
  }
  /**
   * Allocate a device buffer with
   * the same size as the data buffers
   * associated with this gpufort array.
   * \see num_data_bytes()
   * \param[inout] pointer to the buffer to allocate
   */
  __host__ hipError_t gpufort_array2_allocate_device_buffer(
    gpufort::array2<char>* array,
    void** buffer
  ) {
    return array->allocate_device_buffer(buffer);
  }

  /**
   * Deallocate a device buffer
   * created via the allocate_device_buffer routine.
   * \see num_data_bytes(), allocate_device_buffer
   * \param[inout] the buffer to deallocte
   */
  __host__ hipError_t gpufort_array2_deallocate_device_buffer(
      gpufort::array2<char>* array,
      void* buffer) {
    return array->deallocate_device_buffer(buffer);
  }
   
  /**
   * \return size of the array in dimension 'dim'.
   * \param[in] dim selected dimension: 1,...,2
   */
  __host__  int gpufort_array2_size(
      gpufort::array2<char>* array,
      int dim) {
    return array->data.size(dim);
  }
  
  /**
   * \return lower bound (inclusive) of the array in dimension 'dim'.
   * \param[in] dim selected dimension: 1,...,2
   */
  __host__  int gpufort_array2_lbound(
      gpufort::array2<char>* array,
      int dim) {
    return array->data.lbound(dim);
  }
  
  /**
   * \return upper bound (inclusive) of the array in dimension 'dim'.
   * \param[in] dim selected dimension: 1,...,2
   */
  __host__  int gpufort_array2_ubound(
      gpufort::array2<char>* array,
      int dim) {
    return array->data.ubound(dim);
  }

    /**
     * Collapse the array by fixing 1 indices.
     * \return a gpufort array of rank 1.
     * \param[inout] result the collapsed array
     * \param[in]    array  the original array
     * \param[in]    i2,...,i2 indices to fix.
     */
  __host__ void gpufort_array2_collapse_1(
      gpufort::array1<char>* result,
      gpufort::array2<char>* array,
      const int i2
  ) {
    auto collapsed_array = 
      array->collapse( 
        i2);
    result->copy(collapsed_array);
  }
  
  __host__ void gpufort_array2_inc_num_refs(
      gpufort::array2<char>* array
  ) {
    array->num_refs += 1;
  } 
  __host__ hipError_t gpufort_array3_init_async (
      gpufort::array3<char>* array,
      int bytes_per_element,
      char* data_host,
      char* data_dev,
      int* sizes,
      int* lower_bounds,
      hipStream_t stream,
      gpufort::AllocMode alloc_mode,
      gpufort::SyncMode sync_mode
  ) {
    return array->init_async(
      bytes_per_element,
      data_host, data_dev,
      sizes, lower_bounds,
      stream,alloc_mode, sync_mode
    );
  } 
  
  __host__ hipError_t gpufort_array3_destroy_async(
      gpufort::array3<char>* array,
      hipStream_t stream
  ) {
    return array->destroy_async(stream);
  } 
  
  __host__ hipError_t gpufort_array3_copy_to_buffer_async (
      gpufort::array3<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind,
      hipStream_t stream
  ) {
    return array->copy_to_buffer_async(buffer,memcpy_kind,stream);
  } 
  
  __host__ hipError_t gpufort_array3_copy_from_buffer_async (
      gpufort::array3<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind,
      hipStream_t stream
  ) {
    return array->copy_from_buffer_async(buffer,memcpy_kind,stream);
  } 
  
  __host__ hipError_t gpufort_array3_copy_to_host_async (
      gpufort::array3<char>* array,
      hipStream_t stream
  ) {
    return array->copy_to_host_async(stream);
  } 
  
  __host__ hipError_t gpufort_array3_copy_to_device_async (
      gpufort::array3<char>* array,
      hipStream_t stream
  ) {
    return array->copy_to_device_async(stream);
  } 
  
  __host__ hipError_t gpufort_array3_dec_num_refs_async(
      gpufort::array3<char>* array,
      bool destroy_if_zero_refs,
      hipStream_t stream
  ) {
    array->num_refs -= 1;
    if ( destroy_if_zero_refs && array->num_refs == 0 ) {
      return array->destroy_async(stream);
    } {
      return hipSuccess;
    }
  } 
  __host__ hipError_t gpufort_array3_init (
      gpufort::array3<char>* array,
      int bytes_per_element,
      char* data_host,
      char* data_dev,
      int* sizes,
      int* lower_bounds,
      gpufort::AllocMode alloc_mode,
      gpufort::SyncMode sync_mode
  ) {
    return array->init(
      bytes_per_element,
      data_host, data_dev,
      sizes, lower_bounds,
      alloc_mode, sync_mode
    );
  } 
  
  __host__ hipError_t gpufort_array3_destroy(
      gpufort::array3<char>* array
  ) {
    return array->destroy();
  } 
  
  __host__ hipError_t gpufort_array3_copy_to_buffer (
      gpufort::array3<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind
  ) {
    return array->copy_to_buffer(buffer,memcpy_kind);
  } 
  
  __host__ hipError_t gpufort_array3_copy_from_buffer (
      gpufort::array3<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind
  ) {
    return array->copy_from_buffer(buffer,memcpy_kind);
  } 
  
  __host__ hipError_t gpufort_array3_copy_to_host (
      gpufort::array3<char>* array
  ) {
    return array->copy_to_host();
  } 
  
  __host__ hipError_t gpufort_array3_copy_to_device (
      gpufort::array3<char>* array
  ) {
    return array->copy_to_device();
  } 
  
  __host__ hipError_t gpufort_array3_dec_num_refs(
      gpufort::array3<char>* array,
      bool destroy_if_zero_refs
  ) {
    array->num_refs -= 1;
    if ( destroy_if_zero_refs && array->num_refs == 0 ) {
      return array->destroy();
    } {
      return hipSuccess;
    }
  } 

  /**
   * Allocate a host buffer with
   * the same size as the data buffers
   * associated with this gpufort array.
   * \see num_data_bytes()
   * \param[inout] pointer to the buffer to allocate
   * \param[in] pinned If the memory should be pinned (default=true)
   * \param[in] flags  Flags for the host memory allocation (default=0).
   */
  __host__ hipError_t gpufort_array3_allocate_host_buffer(
    gpufort::array3<char>* array,
    void** buffer,bool pinned=true,int flags=0
  ) {
    return array->allocate_host_buffer(buffer,pinned,flags);
  }

  /**
   * Deallocate a host buffer
   * created via the allocate_host_buffer routine.
   * \see num_data_bytes(), allocate_host_buffer
   * \param[inout] the buffer to deallocte
     * \param[in] pinned If the memory to deallocate is pinned [default=true]
   */
  __host__ hipError_t gpufort_array3_deallocate_host_buffer(
      gpufort::array3<char>* array,
      void* buffer,bool pinned=true) {
    return array->deallocate_host_buffer(buffer,pinned);
  }
  /**
   * Allocate a device buffer with
   * the same size as the data buffers
   * associated with this gpufort array.
   * \see num_data_bytes()
   * \param[inout] pointer to the buffer to allocate
   */
  __host__ hipError_t gpufort_array3_allocate_device_buffer(
    gpufort::array3<char>* array,
    void** buffer
  ) {
    return array->allocate_device_buffer(buffer);
  }

  /**
   * Deallocate a device buffer
   * created via the allocate_device_buffer routine.
   * \see num_data_bytes(), allocate_device_buffer
   * \param[inout] the buffer to deallocte
   */
  __host__ hipError_t gpufort_array3_deallocate_device_buffer(
      gpufort::array3<char>* array,
      void* buffer) {
    return array->deallocate_device_buffer(buffer);
  }
   
  /**
   * \return size of the array in dimension 'dim'.
   * \param[in] dim selected dimension: 1,...,3
   */
  __host__  int gpufort_array3_size(
      gpufort::array3<char>* array,
      int dim) {
    return array->data.size(dim);
  }
  
  /**
   * \return lower bound (inclusive) of the array in dimension 'dim'.
   * \param[in] dim selected dimension: 1,...,3
   */
  __host__  int gpufort_array3_lbound(
      gpufort::array3<char>* array,
      int dim) {
    return array->data.lbound(dim);
  }
  
  /**
   * \return upper bound (inclusive) of the array in dimension 'dim'.
   * \param[in] dim selected dimension: 1,...,3
   */
  __host__  int gpufort_array3_ubound(
      gpufort::array3<char>* array,
      int dim) {
    return array->data.ubound(dim);
  }

    /**
     * Collapse the array by fixing 1 indices.
     * \return a gpufort array of rank 2.
     * \param[inout] result the collapsed array
     * \param[in]    array  the original array
     * \param[in]    i3,...,i3 indices to fix.
     */
  __host__ void gpufort_array3_collapse_2(
      gpufort::array2<char>* result,
      gpufort::array3<char>* array,
      const int i3
  ) {
    auto collapsed_array = 
      array->collapse( 
        i3);
    result->copy(collapsed_array);
  }
    /**
     * Collapse the array by fixing 2 indices.
     * \return a gpufort array of rank 1.
     * \param[inout] result the collapsed array
     * \param[in]    array  the original array
     * \param[in]    i2,...,i3 indices to fix.
     */
  __host__ void gpufort_array3_collapse_1(
      gpufort::array1<char>* result,
      gpufort::array3<char>* array,
      const int i2,
      const int i3
  ) {
    auto collapsed_array = 
      array->collapse( 
        i2,i3);
    result->copy(collapsed_array);
  }
  
  __host__ void gpufort_array3_inc_num_refs(
      gpufort::array3<char>* array
  ) {
    array->num_refs += 1;
  } 
  __host__ hipError_t gpufort_array4_init_async (
      gpufort::array4<char>* array,
      int bytes_per_element,
      char* data_host,
      char* data_dev,
      int* sizes,
      int* lower_bounds,
      hipStream_t stream,
      gpufort::AllocMode alloc_mode,
      gpufort::SyncMode sync_mode
  ) {
    return array->init_async(
      bytes_per_element,
      data_host, data_dev,
      sizes, lower_bounds,
      stream,alloc_mode, sync_mode
    );
  } 
  
  __host__ hipError_t gpufort_array4_destroy_async(
      gpufort::array4<char>* array,
      hipStream_t stream
  ) {
    return array->destroy_async(stream);
  } 
  
  __host__ hipError_t gpufort_array4_copy_to_buffer_async (
      gpufort::array4<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind,
      hipStream_t stream
  ) {
    return array->copy_to_buffer_async(buffer,memcpy_kind,stream);
  } 
  
  __host__ hipError_t gpufort_array4_copy_from_buffer_async (
      gpufort::array4<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind,
      hipStream_t stream
  ) {
    return array->copy_from_buffer_async(buffer,memcpy_kind,stream);
  } 
  
  __host__ hipError_t gpufort_array4_copy_to_host_async (
      gpufort::array4<char>* array,
      hipStream_t stream
  ) {
    return array->copy_to_host_async(stream);
  } 
  
  __host__ hipError_t gpufort_array4_copy_to_device_async (
      gpufort::array4<char>* array,
      hipStream_t stream
  ) {
    return array->copy_to_device_async(stream);
  } 
  
  __host__ hipError_t gpufort_array4_dec_num_refs_async(
      gpufort::array4<char>* array,
      bool destroy_if_zero_refs,
      hipStream_t stream
  ) {
    array->num_refs -= 1;
    if ( destroy_if_zero_refs && array->num_refs == 0 ) {
      return array->destroy_async(stream);
    } {
      return hipSuccess;
    }
  } 
  __host__ hipError_t gpufort_array4_init (
      gpufort::array4<char>* array,
      int bytes_per_element,
      char* data_host,
      char* data_dev,
      int* sizes,
      int* lower_bounds,
      gpufort::AllocMode alloc_mode,
      gpufort::SyncMode sync_mode
  ) {
    return array->init(
      bytes_per_element,
      data_host, data_dev,
      sizes, lower_bounds,
      alloc_mode, sync_mode
    );
  } 
  
  __host__ hipError_t gpufort_array4_destroy(
      gpufort::array4<char>* array
  ) {
    return array->destroy();
  } 
  
  __host__ hipError_t gpufort_array4_copy_to_buffer (
      gpufort::array4<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind
  ) {
    return array->copy_to_buffer(buffer,memcpy_kind);
  } 
  
  __host__ hipError_t gpufort_array4_copy_from_buffer (
      gpufort::array4<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind
  ) {
    return array->copy_from_buffer(buffer,memcpy_kind);
  } 
  
  __host__ hipError_t gpufort_array4_copy_to_host (
      gpufort::array4<char>* array
  ) {
    return array->copy_to_host();
  } 
  
  __host__ hipError_t gpufort_array4_copy_to_device (
      gpufort::array4<char>* array
  ) {
    return array->copy_to_device();
  } 
  
  __host__ hipError_t gpufort_array4_dec_num_refs(
      gpufort::array4<char>* array,
      bool destroy_if_zero_refs
  ) {
    array->num_refs -= 1;
    if ( destroy_if_zero_refs && array->num_refs == 0 ) {
      return array->destroy();
    } {
      return hipSuccess;
    }
  } 

  /**
   * Allocate a host buffer with
   * the same size as the data buffers
   * associated with this gpufort array.
   * \see num_data_bytes()
   * \param[inout] pointer to the buffer to allocate
   * \param[in] pinned If the memory should be pinned (default=true)
   * \param[in] flags  Flags for the host memory allocation (default=0).
   */
  __host__ hipError_t gpufort_array4_allocate_host_buffer(
    gpufort::array4<char>* array,
    void** buffer,bool pinned=true,int flags=0
  ) {
    return array->allocate_host_buffer(buffer,pinned,flags);
  }

  /**
   * Deallocate a host buffer
   * created via the allocate_host_buffer routine.
   * \see num_data_bytes(), allocate_host_buffer
   * \param[inout] the buffer to deallocte
     * \param[in] pinned If the memory to deallocate is pinned [default=true]
   */
  __host__ hipError_t gpufort_array4_deallocate_host_buffer(
      gpufort::array4<char>* array,
      void* buffer,bool pinned=true) {
    return array->deallocate_host_buffer(buffer,pinned);
  }
  /**
   * Allocate a device buffer with
   * the same size as the data buffers
   * associated with this gpufort array.
   * \see num_data_bytes()
   * \param[inout] pointer to the buffer to allocate
   */
  __host__ hipError_t gpufort_array4_allocate_device_buffer(
    gpufort::array4<char>* array,
    void** buffer
  ) {
    return array->allocate_device_buffer(buffer);
  }

  /**
   * Deallocate a device buffer
   * created via the allocate_device_buffer routine.
   * \see num_data_bytes(), allocate_device_buffer
   * \param[inout] the buffer to deallocte
   */
  __host__ hipError_t gpufort_array4_deallocate_device_buffer(
      gpufort::array4<char>* array,
      void* buffer) {
    return array->deallocate_device_buffer(buffer);
  }
   
  /**
   * \return size of the array in dimension 'dim'.
   * \param[in] dim selected dimension: 1,...,4
   */
  __host__  int gpufort_array4_size(
      gpufort::array4<char>* array,
      int dim) {
    return array->data.size(dim);
  }
  
  /**
   * \return lower bound (inclusive) of the array in dimension 'dim'.
   * \param[in] dim selected dimension: 1,...,4
   */
  __host__  int gpufort_array4_lbound(
      gpufort::array4<char>* array,
      int dim) {
    return array->data.lbound(dim);
  }
  
  /**
   * \return upper bound (inclusive) of the array in dimension 'dim'.
   * \param[in] dim selected dimension: 1,...,4
   */
  __host__  int gpufort_array4_ubound(
      gpufort::array4<char>* array,
      int dim) {
    return array->data.ubound(dim);
  }

    /**
     * Collapse the array by fixing 1 indices.
     * \return a gpufort array of rank 3.
     * \param[inout] result the collapsed array
     * \param[in]    array  the original array
     * \param[in]    i4,...,i4 indices to fix.
     */
  __host__ void gpufort_array4_collapse_3(
      gpufort::array3<char>* result,
      gpufort::array4<char>* array,
      const int i4
  ) {
    auto collapsed_array = 
      array->collapse( 
        i4);
    result->copy(collapsed_array);
  }
    /**
     * Collapse the array by fixing 2 indices.
     * \return a gpufort array of rank 2.
     * \param[inout] result the collapsed array
     * \param[in]    array  the original array
     * \param[in]    i3,...,i4 indices to fix.
     */
  __host__ void gpufort_array4_collapse_2(
      gpufort::array2<char>* result,
      gpufort::array4<char>* array,
      const int i3,
      const int i4
  ) {
    auto collapsed_array = 
      array->collapse( 
        i3,i4);
    result->copy(collapsed_array);
  }
    /**
     * Collapse the array by fixing 3 indices.
     * \return a gpufort array of rank 1.
     * \param[inout] result the collapsed array
     * \param[in]    array  the original array
     * \param[in]    i2,...,i4 indices to fix.
     */
  __host__ void gpufort_array4_collapse_1(
      gpufort::array1<char>* result,
      gpufort::array4<char>* array,
      const int i2,
      const int i3,
      const int i4
  ) {
    auto collapsed_array = 
      array->collapse( 
        i2,i3,i4);
    result->copy(collapsed_array);
  }
  
  __host__ void gpufort_array4_inc_num_refs(
      gpufort::array4<char>* array
  ) {
    array->num_refs += 1;
  } 
  __host__ hipError_t gpufort_array5_init_async (
      gpufort::array5<char>* array,
      int bytes_per_element,
      char* data_host,
      char* data_dev,
      int* sizes,
      int* lower_bounds,
      hipStream_t stream,
      gpufort::AllocMode alloc_mode,
      gpufort::SyncMode sync_mode
  ) {
    return array->init_async(
      bytes_per_element,
      data_host, data_dev,
      sizes, lower_bounds,
      stream,alloc_mode, sync_mode
    );
  } 
  
  __host__ hipError_t gpufort_array5_destroy_async(
      gpufort::array5<char>* array,
      hipStream_t stream
  ) {
    return array->destroy_async(stream);
  } 
  
  __host__ hipError_t gpufort_array5_copy_to_buffer_async (
      gpufort::array5<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind,
      hipStream_t stream
  ) {
    return array->copy_to_buffer_async(buffer,memcpy_kind,stream);
  } 
  
  __host__ hipError_t gpufort_array5_copy_from_buffer_async (
      gpufort::array5<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind,
      hipStream_t stream
  ) {
    return array->copy_from_buffer_async(buffer,memcpy_kind,stream);
  } 
  
  __host__ hipError_t gpufort_array5_copy_to_host_async (
      gpufort::array5<char>* array,
      hipStream_t stream
  ) {
    return array->copy_to_host_async(stream);
  } 
  
  __host__ hipError_t gpufort_array5_copy_to_device_async (
      gpufort::array5<char>* array,
      hipStream_t stream
  ) {
    return array->copy_to_device_async(stream);
  } 
  
  __host__ hipError_t gpufort_array5_dec_num_refs_async(
      gpufort::array5<char>* array,
      bool destroy_if_zero_refs,
      hipStream_t stream
  ) {
    array->num_refs -= 1;
    if ( destroy_if_zero_refs && array->num_refs == 0 ) {
      return array->destroy_async(stream);
    } {
      return hipSuccess;
    }
  } 
  __host__ hipError_t gpufort_array5_init (
      gpufort::array5<char>* array,
      int bytes_per_element,
      char* data_host,
      char* data_dev,
      int* sizes,
      int* lower_bounds,
      gpufort::AllocMode alloc_mode,
      gpufort::SyncMode sync_mode
  ) {
    return array->init(
      bytes_per_element,
      data_host, data_dev,
      sizes, lower_bounds,
      alloc_mode, sync_mode
    );
  } 
  
  __host__ hipError_t gpufort_array5_destroy(
      gpufort::array5<char>* array
  ) {
    return array->destroy();
  } 
  
  __host__ hipError_t gpufort_array5_copy_to_buffer (
      gpufort::array5<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind
  ) {
    return array->copy_to_buffer(buffer,memcpy_kind);
  } 
  
  __host__ hipError_t gpufort_array5_copy_from_buffer (
      gpufort::array5<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind
  ) {
    return array->copy_from_buffer(buffer,memcpy_kind);
  } 
  
  __host__ hipError_t gpufort_array5_copy_to_host (
      gpufort::array5<char>* array
  ) {
    return array->copy_to_host();
  } 
  
  __host__ hipError_t gpufort_array5_copy_to_device (
      gpufort::array5<char>* array
  ) {
    return array->copy_to_device();
  } 
  
  __host__ hipError_t gpufort_array5_dec_num_refs(
      gpufort::array5<char>* array,
      bool destroy_if_zero_refs
  ) {
    array->num_refs -= 1;
    if ( destroy_if_zero_refs && array->num_refs == 0 ) {
      return array->destroy();
    } {
      return hipSuccess;
    }
  } 

  /**
   * Allocate a host buffer with
   * the same size as the data buffers
   * associated with this gpufort array.
   * \see num_data_bytes()
   * \param[inout] pointer to the buffer to allocate
   * \param[in] pinned If the memory should be pinned (default=true)
   * \param[in] flags  Flags for the host memory allocation (default=0).
   */
  __host__ hipError_t gpufort_array5_allocate_host_buffer(
    gpufort::array5<char>* array,
    void** buffer,bool pinned=true,int flags=0
  ) {
    return array->allocate_host_buffer(buffer,pinned,flags);
  }

  /**
   * Deallocate a host buffer
   * created via the allocate_host_buffer routine.
   * \see num_data_bytes(), allocate_host_buffer
   * \param[inout] the buffer to deallocte
     * \param[in] pinned If the memory to deallocate is pinned [default=true]
   */
  __host__ hipError_t gpufort_array5_deallocate_host_buffer(
      gpufort::array5<char>* array,
      void* buffer,bool pinned=true) {
    return array->deallocate_host_buffer(buffer,pinned);
  }
  /**
   * Allocate a device buffer with
   * the same size as the data buffers
   * associated with this gpufort array.
   * \see num_data_bytes()
   * \param[inout] pointer to the buffer to allocate
   */
  __host__ hipError_t gpufort_array5_allocate_device_buffer(
    gpufort::array5<char>* array,
    void** buffer
  ) {
    return array->allocate_device_buffer(buffer);
  }

  /**
   * Deallocate a device buffer
   * created via the allocate_device_buffer routine.
   * \see num_data_bytes(), allocate_device_buffer
   * \param[inout] the buffer to deallocte
   */
  __host__ hipError_t gpufort_array5_deallocate_device_buffer(
      gpufort::array5<char>* array,
      void* buffer) {
    return array->deallocate_device_buffer(buffer);
  }
   
  /**
   * \return size of the array in dimension 'dim'.
   * \param[in] dim selected dimension: 1,...,5
   */
  __host__  int gpufort_array5_size(
      gpufort::array5<char>* array,
      int dim) {
    return array->data.size(dim);
  }
  
  /**
   * \return lower bound (inclusive) of the array in dimension 'dim'.
   * \param[in] dim selected dimension: 1,...,5
   */
  __host__  int gpufort_array5_lbound(
      gpufort::array5<char>* array,
      int dim) {
    return array->data.lbound(dim);
  }
  
  /**
   * \return upper bound (inclusive) of the array in dimension 'dim'.
   * \param[in] dim selected dimension: 1,...,5
   */
  __host__  int gpufort_array5_ubound(
      gpufort::array5<char>* array,
      int dim) {
    return array->data.ubound(dim);
  }

    /**
     * Collapse the array by fixing 1 indices.
     * \return a gpufort array of rank 4.
     * \param[inout] result the collapsed array
     * \param[in]    array  the original array
     * \param[in]    i5,...,i5 indices to fix.
     */
  __host__ void gpufort_array5_collapse_4(
      gpufort::array4<char>* result,
      gpufort::array5<char>* array,
      const int i5
  ) {
    auto collapsed_array = 
      array->collapse( 
        i5);
    result->copy(collapsed_array);
  }
    /**
     * Collapse the array by fixing 2 indices.
     * \return a gpufort array of rank 3.
     * \param[inout] result the collapsed array
     * \param[in]    array  the original array
     * \param[in]    i4,...,i5 indices to fix.
     */
  __host__ void gpufort_array5_collapse_3(
      gpufort::array3<char>* result,
      gpufort::array5<char>* array,
      const int i4,
      const int i5
  ) {
    auto collapsed_array = 
      array->collapse( 
        i4,i5);
    result->copy(collapsed_array);
  }
    /**
     * Collapse the array by fixing 3 indices.
     * \return a gpufort array of rank 2.
     * \param[inout] result the collapsed array
     * \param[in]    array  the original array
     * \param[in]    i3,...,i5 indices to fix.
     */
  __host__ void gpufort_array5_collapse_2(
      gpufort::array2<char>* result,
      gpufort::array5<char>* array,
      const int i3,
      const int i4,
      const int i5
  ) {
    auto collapsed_array = 
      array->collapse( 
        i3,i4,i5);
    result->copy(collapsed_array);
  }
    /**
     * Collapse the array by fixing 4 indices.
     * \return a gpufort array of rank 1.
     * \param[inout] result the collapsed array
     * \param[in]    array  the original array
     * \param[in]    i2,...,i5 indices to fix.
     */
  __host__ void gpufort_array5_collapse_1(
      gpufort::array1<char>* result,
      gpufort::array5<char>* array,
      const int i2,
      const int i3,
      const int i4,
      const int i5
  ) {
    auto collapsed_array = 
      array->collapse( 
        i2,i3,i4,i5);
    result->copy(collapsed_array);
  }
  
  __host__ void gpufort_array5_inc_num_refs(
      gpufort::array5<char>* array
  ) {
    array->num_refs += 1;
  } 
  __host__ hipError_t gpufort_array6_init_async (
      gpufort::array6<char>* array,
      int bytes_per_element,
      char* data_host,
      char* data_dev,
      int* sizes,
      int* lower_bounds,
      hipStream_t stream,
      gpufort::AllocMode alloc_mode,
      gpufort::SyncMode sync_mode
  ) {
    return array->init_async(
      bytes_per_element,
      data_host, data_dev,
      sizes, lower_bounds,
      stream,alloc_mode, sync_mode
    );
  } 
  
  __host__ hipError_t gpufort_array6_destroy_async(
      gpufort::array6<char>* array,
      hipStream_t stream
  ) {
    return array->destroy_async(stream);
  } 
  
  __host__ hipError_t gpufort_array6_copy_to_buffer_async (
      gpufort::array6<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind,
      hipStream_t stream
  ) {
    return array->copy_to_buffer_async(buffer,memcpy_kind,stream);
  } 
  
  __host__ hipError_t gpufort_array6_copy_from_buffer_async (
      gpufort::array6<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind,
      hipStream_t stream
  ) {
    return array->copy_from_buffer_async(buffer,memcpy_kind,stream);
  } 
  
  __host__ hipError_t gpufort_array6_copy_to_host_async (
      gpufort::array6<char>* array,
      hipStream_t stream
  ) {
    return array->copy_to_host_async(stream);
  } 
  
  __host__ hipError_t gpufort_array6_copy_to_device_async (
      gpufort::array6<char>* array,
      hipStream_t stream
  ) {
    return array->copy_to_device_async(stream);
  } 
  
  __host__ hipError_t gpufort_array6_dec_num_refs_async(
      gpufort::array6<char>* array,
      bool destroy_if_zero_refs,
      hipStream_t stream
  ) {
    array->num_refs -= 1;
    if ( destroy_if_zero_refs && array->num_refs == 0 ) {
      return array->destroy_async(stream);
    } {
      return hipSuccess;
    }
  } 
  __host__ hipError_t gpufort_array6_init (
      gpufort::array6<char>* array,
      int bytes_per_element,
      char* data_host,
      char* data_dev,
      int* sizes,
      int* lower_bounds,
      gpufort::AllocMode alloc_mode,
      gpufort::SyncMode sync_mode
  ) {
    return array->init(
      bytes_per_element,
      data_host, data_dev,
      sizes, lower_bounds,
      alloc_mode, sync_mode
    );
  } 
  
  __host__ hipError_t gpufort_array6_destroy(
      gpufort::array6<char>* array
  ) {
    return array->destroy();
  } 
  
  __host__ hipError_t gpufort_array6_copy_to_buffer (
      gpufort::array6<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind
  ) {
    return array->copy_to_buffer(buffer,memcpy_kind);
  } 
  
  __host__ hipError_t gpufort_array6_copy_from_buffer (
      gpufort::array6<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind
  ) {
    return array->copy_from_buffer(buffer,memcpy_kind);
  } 
  
  __host__ hipError_t gpufort_array6_copy_to_host (
      gpufort::array6<char>* array
  ) {
    return array->copy_to_host();
  } 
  
  __host__ hipError_t gpufort_array6_copy_to_device (
      gpufort::array6<char>* array
  ) {
    return array->copy_to_device();
  } 
  
  __host__ hipError_t gpufort_array6_dec_num_refs(
      gpufort::array6<char>* array,
      bool destroy_if_zero_refs
  ) {
    array->num_refs -= 1;
    if ( destroy_if_zero_refs && array->num_refs == 0 ) {
      return array->destroy();
    } {
      return hipSuccess;
    }
  } 

  /**
   * Allocate a host buffer with
   * the same size as the data buffers
   * associated with this gpufort array.
   * \see num_data_bytes()
   * \param[inout] pointer to the buffer to allocate
   * \param[in] pinned If the memory should be pinned (default=true)
   * \param[in] flags  Flags for the host memory allocation (default=0).
   */
  __host__ hipError_t gpufort_array6_allocate_host_buffer(
    gpufort::array6<char>* array,
    void** buffer,bool pinned=true,int flags=0
  ) {
    return array->allocate_host_buffer(buffer,pinned,flags);
  }

  /**
   * Deallocate a host buffer
   * created via the allocate_host_buffer routine.
   * \see num_data_bytes(), allocate_host_buffer
   * \param[inout] the buffer to deallocte
     * \param[in] pinned If the memory to deallocate is pinned [default=true]
   */
  __host__ hipError_t gpufort_array6_deallocate_host_buffer(
      gpufort::array6<char>* array,
      void* buffer,bool pinned=true) {
    return array->deallocate_host_buffer(buffer,pinned);
  }
  /**
   * Allocate a device buffer with
   * the same size as the data buffers
   * associated with this gpufort array.
   * \see num_data_bytes()
   * \param[inout] pointer to the buffer to allocate
   */
  __host__ hipError_t gpufort_array6_allocate_device_buffer(
    gpufort::array6<char>* array,
    void** buffer
  ) {
    return array->allocate_device_buffer(buffer);
  }

  /**
   * Deallocate a device buffer
   * created via the allocate_device_buffer routine.
   * \see num_data_bytes(), allocate_device_buffer
   * \param[inout] the buffer to deallocte
   */
  __host__ hipError_t gpufort_array6_deallocate_device_buffer(
      gpufort::array6<char>* array,
      void* buffer) {
    return array->deallocate_device_buffer(buffer);
  }
   
  /**
   * \return size of the array in dimension 'dim'.
   * \param[in] dim selected dimension: 1,...,6
   */
  __host__  int gpufort_array6_size(
      gpufort::array6<char>* array,
      int dim) {
    return array->data.size(dim);
  }
  
  /**
   * \return lower bound (inclusive) of the array in dimension 'dim'.
   * \param[in] dim selected dimension: 1,...,6
   */
  __host__  int gpufort_array6_lbound(
      gpufort::array6<char>* array,
      int dim) {
    return array->data.lbound(dim);
  }
  
  /**
   * \return upper bound (inclusive) of the array in dimension 'dim'.
   * \param[in] dim selected dimension: 1,...,6
   */
  __host__  int gpufort_array6_ubound(
      gpufort::array6<char>* array,
      int dim) {
    return array->data.ubound(dim);
  }

    /**
     * Collapse the array by fixing 1 indices.
     * \return a gpufort array of rank 5.
     * \param[inout] result the collapsed array
     * \param[in]    array  the original array
     * \param[in]    i6,...,i6 indices to fix.
     */
  __host__ void gpufort_array6_collapse_5(
      gpufort::array5<char>* result,
      gpufort::array6<char>* array,
      const int i6
  ) {
    auto collapsed_array = 
      array->collapse( 
        i6);
    result->copy(collapsed_array);
  }
    /**
     * Collapse the array by fixing 2 indices.
     * \return a gpufort array of rank 4.
     * \param[inout] result the collapsed array
     * \param[in]    array  the original array
     * \param[in]    i5,...,i6 indices to fix.
     */
  __host__ void gpufort_array6_collapse_4(
      gpufort::array4<char>* result,
      gpufort::array6<char>* array,
      const int i5,
      const int i6
  ) {
    auto collapsed_array = 
      array->collapse( 
        i5,i6);
    result->copy(collapsed_array);
  }
    /**
     * Collapse the array by fixing 3 indices.
     * \return a gpufort array of rank 3.
     * \param[inout] result the collapsed array
     * \param[in]    array  the original array
     * \param[in]    i4,...,i6 indices to fix.
     */
  __host__ void gpufort_array6_collapse_3(
      gpufort::array3<char>* result,
      gpufort::array6<char>* array,
      const int i4,
      const int i5,
      const int i6
  ) {
    auto collapsed_array = 
      array->collapse( 
        i4,i5,i6);
    result->copy(collapsed_array);
  }
    /**
     * Collapse the array by fixing 4 indices.
     * \return a gpufort array of rank 2.
     * \param[inout] result the collapsed array
     * \param[in]    array  the original array
     * \param[in]    i3,...,i6 indices to fix.
     */
  __host__ void gpufort_array6_collapse_2(
      gpufort::array2<char>* result,
      gpufort::array6<char>* array,
      const int i3,
      const int i4,
      const int i5,
      const int i6
  ) {
    auto collapsed_array = 
      array->collapse( 
        i3,i4,i5,i6);
    result->copy(collapsed_array);
  }
    /**
     * Collapse the array by fixing 5 indices.
     * \return a gpufort array of rank 1.
     * \param[inout] result the collapsed array
     * \param[in]    array  the original array
     * \param[in]    i2,...,i6 indices to fix.
     */
  __host__ void gpufort_array6_collapse_1(
      gpufort::array1<char>* result,
      gpufort::array6<char>* array,
      const int i2,
      const int i3,
      const int i4,
      const int i5,
      const int i6
  ) {
    auto collapsed_array = 
      array->collapse( 
        i2,i3,i4,i5,i6);
    result->copy(collapsed_array);
  }
  
  __host__ void gpufort_array6_inc_num_refs(
      gpufort::array6<char>* array
  ) {
    array->num_refs += 1;
  } 
  __host__ hipError_t gpufort_array7_init_async (
      gpufort::array7<char>* array,
      int bytes_per_element,
      char* data_host,
      char* data_dev,
      int* sizes,
      int* lower_bounds,
      hipStream_t stream,
      gpufort::AllocMode alloc_mode,
      gpufort::SyncMode sync_mode
  ) {
    return array->init_async(
      bytes_per_element,
      data_host, data_dev,
      sizes, lower_bounds,
      stream,alloc_mode, sync_mode
    );
  } 
  
  __host__ hipError_t gpufort_array7_destroy_async(
      gpufort::array7<char>* array,
      hipStream_t stream
  ) {
    return array->destroy_async(stream);
  } 
  
  __host__ hipError_t gpufort_array7_copy_to_buffer_async (
      gpufort::array7<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind,
      hipStream_t stream
  ) {
    return array->copy_to_buffer_async(buffer,memcpy_kind,stream);
  } 
  
  __host__ hipError_t gpufort_array7_copy_from_buffer_async (
      gpufort::array7<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind,
      hipStream_t stream
  ) {
    return array->copy_from_buffer_async(buffer,memcpy_kind,stream);
  } 
  
  __host__ hipError_t gpufort_array7_copy_to_host_async (
      gpufort::array7<char>* array,
      hipStream_t stream
  ) {
    return array->copy_to_host_async(stream);
  } 
  
  __host__ hipError_t gpufort_array7_copy_to_device_async (
      gpufort::array7<char>* array,
      hipStream_t stream
  ) {
    return array->copy_to_device_async(stream);
  } 
  
  __host__ hipError_t gpufort_array7_dec_num_refs_async(
      gpufort::array7<char>* array,
      bool destroy_if_zero_refs,
      hipStream_t stream
  ) {
    array->num_refs -= 1;
    if ( destroy_if_zero_refs && array->num_refs == 0 ) {
      return array->destroy_async(stream);
    } {
      return hipSuccess;
    }
  } 
  __host__ hipError_t gpufort_array7_init (
      gpufort::array7<char>* array,
      int bytes_per_element,
      char* data_host,
      char* data_dev,
      int* sizes,
      int* lower_bounds,
      gpufort::AllocMode alloc_mode,
      gpufort::SyncMode sync_mode
  ) {
    return array->init(
      bytes_per_element,
      data_host, data_dev,
      sizes, lower_bounds,
      alloc_mode, sync_mode
    );
  } 
  
  __host__ hipError_t gpufort_array7_destroy(
      gpufort::array7<char>* array
  ) {
    return array->destroy();
  } 
  
  __host__ hipError_t gpufort_array7_copy_to_buffer (
      gpufort::array7<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind
  ) {
    return array->copy_to_buffer(buffer,memcpy_kind);
  } 
  
  __host__ hipError_t gpufort_array7_copy_from_buffer (
      gpufort::array7<char>* array,
      void* buffer,
      hipMemcpyKind memcpy_kind
  ) {
    return array->copy_from_buffer(buffer,memcpy_kind);
  } 
  
  __host__ hipError_t gpufort_array7_copy_to_host (
      gpufort::array7<char>* array
  ) {
    return array->copy_to_host();
  } 
  
  __host__ hipError_t gpufort_array7_copy_to_device (
      gpufort::array7<char>* array
  ) {
    return array->copy_to_device();
  } 
  
  __host__ hipError_t gpufort_array7_dec_num_refs(
      gpufort::array7<char>* array,
      bool destroy_if_zero_refs
  ) {
    array->num_refs -= 1;
    if ( destroy_if_zero_refs && array->num_refs == 0 ) {
      return array->destroy();
    } {
      return hipSuccess;
    }
  } 

  /**
   * Allocate a host buffer with
   * the same size as the data buffers
   * associated with this gpufort array.
   * \see num_data_bytes()
   * \param[inout] pointer to the buffer to allocate
   * \param[in] pinned If the memory should be pinned (default=true)
   * \param[in] flags  Flags for the host memory allocation (default=0).
   */
  __host__ hipError_t gpufort_array7_allocate_host_buffer(
    gpufort::array7<char>* array,
    void** buffer,bool pinned=true,int flags=0
  ) {
    return array->allocate_host_buffer(buffer,pinned,flags);
  }

  /**
   * Deallocate a host buffer
   * created via the allocate_host_buffer routine.
   * \see num_data_bytes(), allocate_host_buffer
   * \param[inout] the buffer to deallocte
     * \param[in] pinned If the memory to deallocate is pinned [default=true]
   */
  __host__ hipError_t gpufort_array7_deallocate_host_buffer(
      gpufort::array7<char>* array,
      void* buffer,bool pinned=true) {
    return array->deallocate_host_buffer(buffer,pinned);
  }
  /**
   * Allocate a device buffer with
   * the same size as the data buffers
   * associated with this gpufort array.
   * \see num_data_bytes()
   * \param[inout] pointer to the buffer to allocate
   */
  __host__ hipError_t gpufort_array7_allocate_device_buffer(
    gpufort::array7<char>* array,
    void** buffer
  ) {
    return array->allocate_device_buffer(buffer);
  }

  /**
   * Deallocate a device buffer
   * created via the allocate_device_buffer routine.
   * \see num_data_bytes(), allocate_device_buffer
   * \param[inout] the buffer to deallocte
   */
  __host__ hipError_t gpufort_array7_deallocate_device_buffer(
      gpufort::array7<char>* array,
      void* buffer) {
    return array->deallocate_device_buffer(buffer);
  }
   
  /**
   * \return size of the array in dimension 'dim'.
   * \param[in] dim selected dimension: 1,...,7
   */
  __host__  int gpufort_array7_size(
      gpufort::array7<char>* array,
      int dim) {
    return array->data.size(dim);
  }
  
  /**
   * \return lower bound (inclusive) of the array in dimension 'dim'.
   * \param[in] dim selected dimension: 1,...,7
   */
  __host__  int gpufort_array7_lbound(
      gpufort::array7<char>* array,
      int dim) {
    return array->data.lbound(dim);
  }
  
  /**
   * \return upper bound (inclusive) of the array in dimension 'dim'.
   * \param[in] dim selected dimension: 1,...,7
   */
  __host__  int gpufort_array7_ubound(
      gpufort::array7<char>* array,
      int dim) {
    return array->data.ubound(dim);
  }

    /**
     * Collapse the array by fixing 1 indices.
     * \return a gpufort array of rank 6.
     * \param[inout] result the collapsed array
     * \param[in]    array  the original array
     * \param[in]    i7,...,i7 indices to fix.
     */
  __host__ void gpufort_array7_collapse_6(
      gpufort::array6<char>* result,
      gpufort::array7<char>* array,
      const int i7
  ) {
    auto collapsed_array = 
      array->collapse( 
        i7);
    result->copy(collapsed_array);
  }
    /**
     * Collapse the array by fixing 2 indices.
     * \return a gpufort array of rank 5.
     * \param[inout] result the collapsed array
     * \param[in]    array  the original array
     * \param[in]    i6,...,i7 indices to fix.
     */
  __host__ void gpufort_array7_collapse_5(
      gpufort::array5<char>* result,
      gpufort::array7<char>* array,
      const int i6,
      const int i7
  ) {
    auto collapsed_array = 
      array->collapse( 
        i6,i7);
    result->copy(collapsed_array);
  }
    /**
     * Collapse the array by fixing 3 indices.
     * \return a gpufort array of rank 4.
     * \param[inout] result the collapsed array
     * \param[in]    array  the original array
     * \param[in]    i5,...,i7 indices to fix.
     */
  __host__ void gpufort_array7_collapse_4(
      gpufort::array4<char>* result,
      gpufort::array7<char>* array,
      const int i5,
      const int i6,
      const int i7
  ) {
    auto collapsed_array = 
      array->collapse( 
        i5,i6,i7);
    result->copy(collapsed_array);
  }
    /**
     * Collapse the array by fixing 4 indices.
     * \return a gpufort array of rank 3.
     * \param[inout] result the collapsed array
     * \param[in]    array  the original array
     * \param[in]    i4,...,i7 indices to fix.
     */
  __host__ void gpufort_array7_collapse_3(
      gpufort::array3<char>* result,
      gpufort::array7<char>* array,
      const int i4,
      const int i5,
      const int i6,
      const int i7
  ) {
    auto collapsed_array = 
      array->collapse( 
        i4,i5,i6,i7);
    result->copy(collapsed_array);
  }
    /**
     * Collapse the array by fixing 5 indices.
     * \return a gpufort array of rank 2.
     * \param[inout] result the collapsed array
     * \param[in]    array  the original array
     * \param[in]    i3,...,i7 indices to fix.
     */
  __host__ void gpufort_array7_collapse_2(
      gpufort::array2<char>* result,
      gpufort::array7<char>* array,
      const int i3,
      const int i4,
      const int i5,
      const int i6,
      const int i7
  ) {
    auto collapsed_array = 
      array->collapse( 
        i3,i4,i5,i6,i7);
    result->copy(collapsed_array);
  }
    /**
     * Collapse the array by fixing 6 indices.
     * \return a gpufort array of rank 1.
     * \param[inout] result the collapsed array
     * \param[in]    array  the original array
     * \param[in]    i2,...,i7 indices to fix.
     */
  __host__ void gpufort_array7_collapse_1(
      gpufort::array1<char>* result,
      gpufort::array7<char>* array,
      const int i2,
      const int i3,
      const int i4,
      const int i5,
      const int i6,
      const int i7
  ) {
    auto collapsed_array = 
      array->collapse( 
        i2,i3,i4,i5,i6,i7);
    result->copy(collapsed_array);
  }
  
  __host__ void gpufort_array7_inc_num_refs(
      gpufort::array7<char>* array
  ) {
    array->num_refs += 1;
  } 
} // extern "C"