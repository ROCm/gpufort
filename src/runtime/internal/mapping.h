#ifndef __INTERNAL_MAPPING_H__
#define __INTERNAL_MAPPING_H__
#endif // __INTERNAL_MAPPING_H__

#include <cstddef>

#include "definitions.h"

namespace internal {
  class Mapping;
}

class internal::Mapping {
private:
  GpufortMapKind kind        = GPUFORT_MAP_UNDEFINED;
  int            id          = -1;
  char*          key_ptr     = nullptr; //!< Usually a host ptr but might be a device pointer as well depending
                                        //!< on the mapping type.
  char*          device_ptr  = nullptr; //!< A device pointer or nullptr;
  char*          interop_ptr = nullptr; //!< A host pointer used for interoperable "shadow" structs created by GPUFORT.
  std::size_t    num_bytes   = 0;       //!< Number of bytes(of the objects at device_ptr and/or interop_ptr address).
  
  /**
   * This counter is incremented when entering each data or compute region that contain an
   * explicit data clause or implicity-determined data attributes for that 
   * block of memory.
   *
   * @note: The reference counters are modified synchronously with the local thread,
   * even if the data directives include an async clause.
   */
  int    structured_ref_ctr = 0; 
  /**
   * This counter is incremented for each enter data copyin/create clause,
   * or each acc_copyin or acc_create API routine call for that block of 
   * memory.
   * 
   * This dynamic reference counter is decrementd for each exit data copyout/delete API routine call
   * for that block of memory. The dynamic reference counter will be set to zero 
   * with an exit data copyout or delete clause when a finalize clause appears
   * or each acc_copyout_finalize/acc_delete_finalize API routine call for
   * the block of memory.
   *
   * @note: The reference couters are modified synchronously with the local thread,
   * even if the data directives include an async clause.
   * 
   *
   */
  int    dynamic_ref_ctr    = 0;
  //int    attachment_ctr     = 0; // not supported
  
  /**
   * @return if a memory region contains another subregion.
   * @param[in] region           address of the memory region's first element (base)
   * @param[in] bytes            size of the memory region (in bytes)
   * @param[in] subregion        address of the memory subregion's first element (base)
   * @param[in] bytes_subregion  size of the subregion (in bytes)
   */
  bool contains_subregion(char* region, std::size_t bytes, char* subregion, std::size_t bytes_subregion);
public:
  void print();
  bool is_initialized();
  bool is_subarray(

  );
  void initialize(
    int id, char* key_ptr, char* device_ptr, char* interop_ptr, 
    std::size_t num_bytes, GpufortMapKind map_kind
  );
  void destroy();
  void copy_to_device();
  void copy_to_host  ();
  void inc_structured_ref_ctr();
  void dec_structured_ref_ctr();
  bool inc_dynamic_ref_ctr();
  bool dec_dynamic_ref_ctr();
};
