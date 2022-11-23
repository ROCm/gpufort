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
      case acc_property_free_memory:
        return gpufortrt_property_free_memory;
      case acc_property_shared_memory_support:
        return gpufortrt_property_shared_memory_support;
      case acc_property_name:
        return gpufortrt_property_name;
      case acc_property_vendor:
        return gpufortrt_property_vendor;
      case acc_property_driver:
        return gpufortrt_property_driver;
    }
  }

  acc_device_t check_device_type(acc_device_t dev_type,bool allow_none=true) {
    switch (dev_type) {
      case acc_device_default:
        return ::default_device;
      case acc_device_current:
        return ::current_device;
      case acc_device_host:
        return acc_device_host;
      case acc_device_not_host: 
      case acc_device_hip: 
      case acc_device_radeon: 
      case acc_device_nvidia:
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
    return gpufortrt_get_num_devices();
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
size_t acc_get_property_f(int dev_num,
                        acc_device_t dev_type,
                        acc_device_property_t property) {
  if ( ::check_device_type(dev_type) == acc_device_hip ) {
    return gpufortrt_get_property_f(dev_num,check_device_property(property));
  } else {
    throw std::invalid_argument("acc_get_property: only implemented for non-host device types");
  }
}

const 
char* acc_get_property_string_f(int dev_num,
                              acc_device_t dev_type,
                              acc_device_property_t property) {
  if ( ::check_device_type(dev_type) == acc_device_hip ) {
    return gpufortrt_get_property_string_f(dev_num,check_device_property(property));
  } else {
    throw std::invalid_argument("acc_get_property_string: only implemented for non-host device types");
  }
}

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
  if(gpufortrt_is_present(hostptr, num_bytes)) 
    return 1;
  else 
    return 0;
}

void acc_wait(int wait_arg){
  gpufortrt_wait(&wait_arg, 1, true);
}

void acc_wait_async(int wait_arg, int async_arg){
  gpufortrt_wait_async(&wait_arg, 1, &async_arg, 1, true);
}

void acc_wait_all(){
  gpufortrt_wait_all(true);
}

void acc_wait_all_async(int async_arg){
  gpufortrt_wait_all_async(&async_arg, 1, true);
}

void acc_wait_device(int wait_arg, int dev_num){
  gpufortrt_wait_device(&wait_arg, 1, dev_num, true);
}

void acc_wait_device_async(int wait_arg, int async_arg, int dev_num){
  gpufortrt_wait_device_async(&wait_arg, 1, &async_arg, 1, dev_num, true);
}

void acc_wait_all_device(int dev_num){
  gpufortrt_wait_all_device(dev_num, true);
}

void acc_wait_all_device_async(int async_arg, int dev_num){
  gpufortrt_wait_all_device_async(&async_arg, 1, dev_num, true);
}

int acc_async_test(int wait_arg){
  return gpufortrt_async_test(wait_arg);
}

int acc_async_test_device(int wait_arg, int dev_num){
  return gpufortrt_async_test_device( wait_arg, dev_num);
}

int acc_async_test_all(void){
  return gpufortrt_async_test_all();
}

int acc_async_test_all_device(int dev_num){
  return gpufortrt_async_test_all_device(dev_num);
}

d_void* acc_copyin(h_void* data_arg, size_t bytes){
  return gpufortrt_copyin(data_arg, bytes, true);
}

void acc_copyin_async(h_void* data_arg, size_t bytes,
                      int async_arg){
  gpufortrt_copyin_async(data_arg, bytes, async_arg, false);
}

d_void* acc_create(h_void* data_arg, size_t bytes){
  return gpufortrt_create(data_arg, bytes, false);
}

void acc_create_async(h_void* data_arg, size_t bytes,
                      int async_arg){
  gpufortrt_create_async(data_arg, bytes, async_arg, false);
}

void acc_copyout(h_void* data_arg, size_t bytes){
  gpufortrt_copyout(data_arg, bytes);
}
void acc_copyout_async(h_void* data_arg, size_t bytes, 
                        int async_arg){
  gpufortrt_copyout_async(data_arg, bytes, async_arg);
}
void acc_copyout_finalize(h_void* data_arg, size_t bytes){
  gpufortrt_copyout_finalize(data_arg, bytes);
}
void acc_copyout_finalize_async(h_void* data_arg, size_t bytes, 
                                int async_arg){
  gpufortrt_copyout_finalize_async( data_arg, bytes, async_arg);
}

int acc_get_default_async(void){
  return gpufortrt_get_default_async();
}

void acc_set_default_async(int async_arg){
  gpufortrt_set_default_async(async_arg);
}

void acc_delete(h_void* data_arg, size_t bytes){
  gpufortrt_delete( data_arg, bytes);
}

void acc_delete_async(h_void* data_arg, size_t bytes,
                      int async_arg){
  gpufortrt_delete_async(data_arg, bytes, async_arg);
}

void acc_delete_finalize(h_void* data_arg, size_t bytes){
  gpufortrt_delete_finalize( data_arg, bytes);
}

void acc_delete_finalize_async(h_void* data_arg,
                               size_t bytes, int async_arg){
  gpufortrt_delete_finalize_async(data_arg, bytes, async_arg);
}

void acc_update_device(h_void* data_arg, size_t bytes){
  gpufortrt_update_device(data_arg, bytes, true, false);
}

void acc_update_device_async(h_void* data_arg, size_t bytes,
                             int async_arg){
  gpufortrt_update_device_async(data_arg, bytes, true, false, async_arg);
}

void acc_update_self(h_void* data_arg, size_t bytes){
  gpufortrt_update_self(data_arg, bytes, true, false);
}

void acc_update_self_async(h_void* data_arg, size_t bytes,
                           int async_arg){
  gpufortrt_update_self_async(data_arg, bytes, true, false, async_arg);
}

d_void* acc_deviceptr(h_void* data_arg){
  return gpufortrt_deviceptr(data_arg);
}

d_void* acc_malloc(size_t bytes){
  h_void* hostptr = nullptr;
  d_void* deviceptr = nullptr; 
  hostptr = (h_void*) malloc(bytes);
  deviceptr = gpufortrt_create(hostptr, bytes, false);
  free(hostptr);
  return deviceptr;
}