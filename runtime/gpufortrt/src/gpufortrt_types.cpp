#include "gpufortrt_types.h"
int gpufortrt_async_noval = -1;
int gpufortrt_async_sync = -2;
int gpufortrt_async_default = -3;

gpufortrt_queue_t gpufortrt_default_queue = nullptr;

#ifdef __cplusplus
std::ostream& operator<<(std::ostream& os, gpufortrt_map_kind_t map_kind) {
  switch(map_kind) {
    case gpufortrt_map_kind_undefined      : os << "undefined"; break;
    case gpufortrt_map_kind_present        : os << "present"; break;
    case gpufortrt_map_kind_delete         : os << "delete"; break;
    case gpufortrt_map_kind_create         : os << "create"; break;
    case gpufortrt_map_kind_no_create      : os << "no_create"; break;
    case gpufortrt_map_kind_copyin         : os << "copyin"; break;
    case gpufortrt_map_kind_copyout        : os << "copyout"; break;
    case gpufortrt_map_kind_copy           : os << "copy"; break;
    default: throw std::invalid_argument("operator<<: invalid value for `map_kind`");
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, gpufortrt_counter_t counter) {
  switch(counter) {
    case gpufortrt_counter_none       : os << "none"; break;
    case gpufortrt_counter_structured : os << "structured"; break;
    case gpufortrt_counter_dynamic    : os << "dynamic"; break;
    default: throw std::invalid_argument("operator<<: invalid value for `counter`");
  }
  return os;
}

std::ostream& operator<<(std::ostream& os,const gpufortrt_mapping_t& mapping) {
  os << "hostptr:"           << mapping.hostptr  
     << ", num_bytes:"       << mapping.num_bytes 
     << ", map_kind:"        << static_cast<gpufortrt_map_kind_t>(mapping.map_kind)
     << ", never_deallocate:" << mapping.never_deallocate;
  return os;
}
#endif