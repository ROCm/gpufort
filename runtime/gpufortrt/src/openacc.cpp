#include "openacc.h"
#include "gpufortrt_api.h"
#include "auxiliary.h"

int acc_async_noval = -1;
int acc_async_sync = -2;
int acc_async_default = -3;

namespace {
  acc_device_t default_device = acc_device_hip;

  acc_device_t current_device = acc_device_none;

  gpufortrt_device_property_t check_device_property(acc_device_property_t property,bool allow_none=true){
    switch(property){
      case acc_property_memory:
        return gpufortrt_property_memory;
      case gpufortrt_property_free_memory:
        return gpufortrt_property_free_memory;
      case gpufortrt_property_shared_memory_support:
        return gpufortrt_property_shared_memory_support;
      case gpufortrt_property_name:
        return gpufortrt_property_name;
      case gpufortrt_property_vendor:
        return gpufortrt_property_vendor;
      case gpufortrt_property_driver:
        return gpufortrt_property_driver;
    }
  }

  bool check_device_type(acc_device_t dev_type,bool allow_none=true) {
    switch (dev_type) {
      case acc_device_default:
        return ::check_device_type(::default_device);
      case acc_device_current:
        return ::check_device_type(::current_device);
      case acc_device_host:
        return acc_device_host;
      case acc_device_not_host: 
      // According to the definition of enum acc_device_t 
      // the following cases are identical. So, keep one? 
      // case acc_device_not_host: 
      // case acc_device_hip: 
      // case acc_device_radeon: 
      // case acc_device_nvidia:
        return acc_device_hip;
      case acc_device_none:
        if ( !allow_none ) {
          throw std::invalid_argument("'acc_device_none' not allowed");
        }
        return acc_device_none;
      default:
        throw std::invalid_argument("unexpected value for 'dev_type'");
        break;
    }
  }
} // namespace {

void acc_set_device_type(acc_device_t dev_type) {
  ::current_device = dev_type;
}

acc_device_t acc_get_device_type(void) {
  return ::current_device;
}

int acc_get_num_devices(acc_device_t dev_type) {
  if ( ::check_device_type(dev_type) == acc_device_hip ) {
    return gpufortrt_get_device_num();
  } else { // host
    return 1;
  }
}


void acc_set_device_num(int dev_num, acc_device_t dev_type) {
  // OpenACC 3.1, Section 3.2.4. Description:
  // "[...]If the value of dev_num is negative, the runtime will revert to its default behavior,
  // which is implementation-defined. If the value of the dev_type is zero, 
  // the selected device number will be used for all device types.
  // Calling acc_set_device_num implies a call to acc_set_device_type(dev_type)"
  if ( static_cast<int>(dev_type) <= 0 || 
       ::check_device_type(dev_type) == acc_device_hip ) {
    gpufortrt_set_device_num(dev_num);
  }
}

int acc_get_device_num(acc_device_t dev_type) {
  if ( ::check_device_type(dev_type) == acc_device_hip ) {
    return gpufortrt_get_device_num();
  } else { // host
    return 0;
  }
}

size_t acc_get_property(int dev_num,
                        acc_device_t dev_type,
                        acc_device_property_t property) {
  if ( ::check_device_type(dev_type) == acc_device_hip ) {
    return gpufortrt_get_property(dev_num,check_device_property(property));
  } else {
    throw std::invalid_argument("acc_get_property: only implemented for non-host device types");
  }
}
const 
char* acc_get_property_string(int dev_num,
                              acc_device_t dev_type,
                              acc_device_property_t property) {
  if ( ::check_device_type(dev_type) == acc_device_hip ) {
    return gpufortrt_get_property_string(dev_num,check_device_property(property));
  } else {
    throw std::invalid_argument("acc_get_property_string: only implemented for non-host device types");
  }
}

// Explicit Fortran interfaces that assume device number starts from 1
void acc_set_device_num_f(int dev_num, acc_device_t dev_type) {
  acc_set_device_num(dev_num-1,dev_type);
}
int acc_get_device_num_f(acc_device_t dev_type) {
  return acc_get_device_num(dev_type)+1;
}


void acc_init(acc_device_t dev_type) {
  if ( ::check_device_type(dev_type) == acc_device_hip ) {
    gpufortrt_init();
  }
}

void acc_shutdown(acc_device_t dev_type) {
  if ( ::check_device_type(dev_type) == acc_device_hip ) {
    gpufortrt_shutdown();
  }
}

int acc_is_present(void* hostptr,std::size_t num_bytes) {
  if(gpufortrt_present(hostptr, num_bytes)) 
    return 1;
  else 
    return 0;
}

void acc_wait_device(int* wait_arg, int num_wait, int dev_num, bool if_arg){
  gpufortrt_wait_device(wait_arg, num_wait, dev_num, if_arg);
}

void acc_wait_device_async(int* wait_arg, int num_wait, 
                           int* async_arg, int num_async,
                           int dev_num, bool if_arg){
  gpufortrt_wait_device_async(wait_arg, num_wait, async_arg, num_async, dev_num, if_arg);
}

void acc_wait_all_device(int dev_num, bool if_arg){
  gpufortrt_wait_all_device(dev_num, if_arg);
}

void acc_wait_all_device_async(int* async_arg, int num_async,
                           int dev_num, bool if_arg){
  gpufortrt_wait_all_device_async(async_arg, num_async, dev_num, if_arg);
}