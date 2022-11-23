// SPDX-License-Identifier: MIT
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "gpufortrt_api.h"

#include <string>
#include <sstream>

#include "hip/hip_runtime_api.h"

#include "internal/auxiliary.h"
#include "internal/gpufortrt_core.h"
  
int gpufortrt_get_default_async(void) {
  return gpufortrt::internal::default_async_arg;  
}

void gpufortrt_set_default_async(int async_arg) {
  gpufortrt::internal::default_async_arg = async_arg;
}

void gpufortrt_set_device_num(int dev_num) {
  HIP_CHECK(hipSetDevice(dev_num)) // TODO backend specific, externalize
}

int gpufortrt_get_num_devices() {
  int dev_num;
  HIP_CHECK(hipGetDeviceCount(&dev_num))
  return dev_num;
}

int gpufortrt_get_device_num() {
  int dev_num;
  HIP_CHECK(hipGetDevice(&dev_num))
  return dev_num;
}

size_t gpufortrt_get_property(int dev_num,
                              gpufortrt_device_property_t property) {
  size_t free, total;
  switch ( property ) {
    case gpufortrt_property_memory:
      HIP_CHECK(hipMemGetInfo(&free, &total))
      return total;
      break;
    case gpufortrt_property_free_memory:
      HIP_CHECK(hipMemGetInfo(&free, &total))
      return free;
      break;
    case gpufortrt_property_shared_memory_support:
      int result;
      HIP_CHECK(hipDeviceGetAttribute(&result,hipDeviceAttributeManagedMemory,dev_num))
      return result;
      break;
    default:
      throw std::invalid_argument("gpufortrt_get_property: property type must be 'gpufortrt_property_memory', 'gpufortrt_free_memory', or 'gpufortrt_property_shared_memory_support'");
      break;
  }
}

const 
char* gpufortrt_get_property_string(int dev_num,
                                    gpufortrt_device_property_t property) {
      throw std::invalid_argument("gpufortrt_get_property_string: not implemented"); // TODO implement
}

size_t gpufortrt_get_property_f(int dev_num,
                                gpufortrt_device_property_t property) {
  return gpufortrt_get_property(dev_num-1,property);
}
const 
char* gpufortrt_get_property_string_f(int dev_num,
                                      gpufortrt_device_property_t property) {
  return gpufortrt_get_property_string(dev_num-1,property);
}

// Explicit Fortran interfaces that assume device number starts from 1
void gpufortrt_set_device_num_f(int dev_num) {
  gpufortrt_set_device_num(dev_num-1);
}
int gpufortrt_get_device_num_f() {
  return gpufortrt_get_device_num()+1;
}

void gpufortrt_mapping_init(
    gpufortrt_mapping_t* mapping,
    void* hostptr,
    std::size_t num_bytes,
    gpufortrt_map_kind_t map_kind,
    bool never_deallocate) {
  mapping->hostptr = hostptr;
  mapping->num_bytes = num_bytes;
  mapping->map_kind = map_kind;
  mapping->never_deallocate = never_deallocate;
}

void gpufortrt_init() {
  if ( gpufortrt::internal::initialized ) {
    throw std::invalid_argument("init: runtime has already been initialized");
  } else {
    gpufortrt::internal::set_from_environment(gpufortrt::internal::LOG_LEVEL,"GPUFORTRT_LOG_LEVEL");
    gpufortrt::internal::set_from_environment(gpufortrt::internal::INITIAL_RECORDS_CAPACITY,"GPUFORTRT_INITIAL_RECORDS_CAPACITY");
    gpufortrt::internal::set_from_environment(gpufortrt::internal::INITIAL_QUEUE_RECORDS_CAPACITY,"GPUFORTRT_INITIAL_QUEUE_RECORDS_CAPACITY");
    gpufortrt::internal::set_from_environment(gpufortrt::internal::INITIAL_STRUCTURED_REGION_STACK_CAPACITY,"GPUFORTRT_INITIAL_STRUCTURED_REGION_STACK_CAPACITY");
    gpufortrt::internal::set_from_environment(gpufortrt::internal::BLOCK_SIZE,"GPUFORTRT_BLOCK_SIZE");
    gpufortrt::internal::set_from_environment(gpufortrt::internal::REUSE_THRESHOLD,"GPUFORTRT_REUSE_THRESHOLD");
    gpufortrt::internal::set_from_environment(gpufortrt::internal::NUM_REFS_TO_DEALLOCATE,"GPUFORTRT_NUM_REFS_TO_DEALLOCATE");
    gpufortrt::internal::record_list.reserve(gpufortrt::internal::INITIAL_RECORDS_CAPACITY);
    gpufortrt::internal::queue_record_list.reserve(gpufortrt::internal::INITIAL_QUEUE_RECORDS_CAPACITY);
    gpufortrt::internal::structured_region_stack.reserve(gpufortrt::internal::INITIAL_STRUCTURED_REGION_STACK_CAPACITY);
    gpufortrt::internal::initialized = true;
    //
    LOG_INFO(1,"init;")
    LOG_INFO(1,".GPUFORTRT_LOG_LEVEL=" << gpufortrt::internal::LOG_LEVEL)
    LOG_INFO(1,".GPUFORTRT_INITIAL_RECORDS_CAPACITY=" << gpufortrt::internal::INITIAL_RECORDS_CAPACITY)
    LOG_INFO(1,".GPUFORTRT_INITIAL_QUEUE_RECORDS_CAPACITY=" << gpufortrt::internal::INITIAL_QUEUE_RECORDS_CAPACITY)
    LOG_INFO(1,".GPUFORTRT_INITIAL_STRUCTURED_REGION_STACK_CAPACITY=" << gpufortrt::internal::INITIAL_STRUCTURED_REGION_STACK_CAPACITY)
    LOG_INFO(1,".GPUFORTRT_BLOCK_SIZE=" << gpufortrt::internal::BLOCK_SIZE)
    LOG_INFO(1,".GPUFORTRT_REUSE_THRESHOLD=" << gpufortrt::internal::REUSE_THRESHOLD)
    LOG_INFO(1,".GPUFORTRT_NUM_REFS_TO_DEALLOCATE=" << gpufortrt::internal::NUM_REFS_TO_DEALLOCATE)
  }
}

void gpufortrt_shutdown() {
  LOG_INFO(1,"shutdown;")
  gpufortrt_wait_all(true);
  if ( !gpufortrt::internal::initialized ) {
    LOG_ERROR("gpufortrt_shutdown: runtime has not been initialized")
  }
  gpufortrt::internal::record_list.destroy();
  gpufortrt::internal::queue_record_list.destroy();
}
  
namespace gpufortrt {
  namespace internal {
      std::tuple<bool/*blocking*/,int/*async_arg*/> check_async_arg(const int async_arg) {
      if ( async_arg >= gpufortrt_async_sync ) {
        return std::make_tuple(
                async_arg == gpufortrt_async_sync/*blocking*/,
                async_arg);
      } else {
        int default_async_arg = gpufortrt::internal::default_async_arg;
        return std::make_tuple(
                default_async_arg == gpufortrt_async_sync/*blocking*/,
                default_async_arg);
      }
    }
  }
}

namespace {

  void* no_create_action(const gpufortrt_counter_t ctr_to_update,
                         void* hostptr,
                         std::size_t num_bytes) {
    LOG_INFO(1, "no_create;")
    if ( !gpufortrt::internal::initialized ) LOG_ERROR("no_create_action: runtime not initialized")
    if ( hostptr != nullptr ) { // nullptr means no-op
      auto inc_tuple/*present,loc*/ = gpufortrt::internal::record_list.increment_record_if_present(
              ctr_to_update,hostptr,num_bytes,false/*check ...*/);
      const bool& success = std::get<0>(inc_tuple);
      const std::size_t& loc = std::get<1>(inc_tuple); 
      if ( success ) { 
        auto& record = gpufortrt::internal::record_list[loc];
        if ( ctr_to_update == gpufortrt_counter_structured ) {
          gpufortrt::internal::structured_region_stack.push_back(
                  gpufortrt_map_kind_no_create,&record);
        }
        return gpufortrt::internal::offsetted_record_deviceptr(record,hostptr);
      } else {
        if ( ctr_to_update == gpufortrt_counter_structured ) {
          gpufortrt::internal::structured_region_stack.push_back(
                  gpufortrt_map_kind_no_create,hostptr,num_bytes);
        }
        return hostptr;
      }
    } else {
      return nullptr;
    }
  }

  void* create_increment_action(
      const gpufortrt_counter_t ctr_to_update,
      void* hostptr,
      const std::size_t num_bytes,
      const gpufortrt_map_kind_t map_kind,
      const bool never_deallocate,
      const bool blocking,
      const int async_arg) {
    if ( ! gpufortrt::internal::initialized ) gpufortrt_init();
    LOG_INFO(1, map_kind
             << ((!blocking) ? " async" : "")
             << "; ctr_to_update:" << ctr_to_update
             << ", hostptr:" << hostptr 
             << ", num_bytes:" << num_bytes 
             << ", never_deallocate:" << never_deallocate 
             << ((!blocking) ? ", async_arg:" : "")
             << ((!blocking) ? std::to_string(async_arg).c_str() : ""))
    if ( hostptr == nullptr) { // nullptr means no-op
      return nullptr;
    } else {
      gpufortrt_queue_t queue = gpufortrt_default_queue;
      if ( !blocking ) {
        queue = gpufortrt::internal::queue_record_list.use_create_queue(async_arg);
      }
      std::size_t loc = gpufortrt::internal::record_list.create_increment_record(
        ctr_to_update,
        hostptr,
        num_bytes,
        never_deallocate, 
        gpufortrt::internal::implies_allocate_device_buffer(map_kind,ctr_to_update),
        gpufortrt::internal::implies_copy_to_device(map_kind),
        blocking,
        queue);

      auto& record = gpufortrt::internal::record_list[loc];
      if ( ctr_to_update == gpufortrt_counter_structured ) {
        gpufortrt::internal::structured_region_stack.push_back(
                map_kind,&record);
      }
      return gpufortrt::internal::offsetted_record_deviceptr(record,hostptr);
    }
  }
  
  void decrement_release_action(void* hostptr,
                                const std::size_t num_bytes,
                                const gpufortrt_map_kind_t map_kind,
                                const bool finalize,
                                const bool blocking,
                                const int async_arg) {
    bool copyout = gpufortrt::internal::implies_copy_to_host(map_kind);
    LOG_INFO(1, ((copyout) ? "copyout" : "delete")
            << ((finalize) ? " finalize" : "")
            << ((!blocking) ? " async" : "")
            << "; hostptr:"<<hostptr 
            << ((!blocking) ? ", async_arg:" : "")
            << ((!blocking) ? std::to_string(async_arg).c_str() : ""))
    if ( !gpufortrt::internal::initialized ) LOG_ERROR("decrement_release_action: runtime not initialized")
    if ( hostptr != nullptr ) { // nullptr means no-op
      gpufortrt_queue_t queue = gpufortrt_default_queue;
      if ( !blocking ) {
        queue = gpufortrt::internal::queue_record_list.use_create_queue(async_arg);
      }
      gpufortrt::internal::record_list.decrement_release_record(
        gpufortrt_counter_dynamic,
        hostptr,
        num_bytes,
        copyout,
        finalize,
        blocking,
        queue);
    }
  } // namespace

  void apply_mappings(gpufortrt_mapping_t* mappings,
                      int num_mappings,
                      gpufortrt_counter_t ctr_to_update,
                      bool blocking,
                      int async_arg,bool finalize) {
    LOG_INFO(1,((ctr_to_update == gpufortrt_counter_structured) ? "data start" : "enter/exit data") 
             << "; num_mappings:" << num_mappings 
             << ", blocking:" << blocking
             << ", async_arg:" << async_arg
             << ", finalize:" << finalize)
    if ( gpufortrt::internal::LOG_LEVEL > 0 ) { 
      for (int i = 0; i < num_mappings; i++) {
         auto mapping = mappings[i];
         LOG_INFO(1,"  mapping "<<i<<": "<<mapping) 
      }
    }
    for (int i = 0; i < num_mappings; i++) {
      auto mapping = mappings[i];

      switch (mapping.map_kind) {
        case gpufortrt_map_kind_delete:
           ::decrement_release_action(
               mapping.hostptr,
               mapping.num_bytes,
               mapping.map_kind,
               finalize,
               blocking,
               async_arg);
           break;
        case gpufortrt_map_kind_copyout:
           switch ( ctr_to_update ) {
             case gpufortrt_counter_structured:
               ::create_increment_action(
                 ctr_to_update, 
                 mapping.hostptr,
                 mapping.num_bytes,
                 mapping.map_kind,
                 mapping.never_deallocate, 
                 blocking,
                 async_arg);
               break;
             case gpufortrt_counter_dynamic:
               ::decrement_release_action(
                   mapping.hostptr,
                   mapping.num_bytes,
                   mapping.map_kind,
                   finalize,
                   blocking,
                   async_arg);
               break;
             default: std::invalid_argument("apply_mappings: invalid ctr_to_update"); break;
           }
           break;
        case gpufortrt_map_kind_no_create:
           ::no_create_action(
             ctr_to_update,
             mapping.hostptr,
             mapping.num_bytes);
           break;
        case gpufortrt_map_kind_present:
        case gpufortrt_map_kind_create:
        case gpufortrt_map_kind_copyin:
        case gpufortrt_map_kind_copy:
           ::create_increment_action(
             ctr_to_update,
             mapping.hostptr,
             mapping.num_bytes,
             mapping.map_kind,
             mapping.never_deallocate, 
             blocking,
             async_arg);
           break;
        default: std::invalid_argument("apply_mappings: invalid map_kind"); break;
      }
    }
  }
} // namespace

void gpufortrt_data_start(gpufortrt_mapping_t* mappings,int num_mappings) {
  if ( !gpufortrt::internal::initialized ) gpufortrt_init();
  gpufortrt::internal::structured_region_stack.enter_structured_region();
  ::apply_mappings(mappings,
                   num_mappings,
                   gpufortrt_counter_structured,
                   true,gpufortrt_async_noval,false/*finalize*/);
}

void gpufortrt_data_end() {
  LOG_INFO(1,"data end;") 
  if ( !gpufortrt::internal::initialized ) LOG_ERROR("gpufortrt_data_end: runtime not initialized")
  gpufortrt::internal::structured_region_stack.leave_structured_region(false,nullptr);
}

void gpufortrt_data_start_async(gpufortrt_mapping_t* mappings,int num_mappings,int async_arg) {
  if ( !gpufortrt::internal::initialized ) gpufortrt_init();
  gpufortrt::internal::structured_region_stack.enter_structured_region();
  
  bool blocking; int async_val;
  std::tie(blocking,async_val) = gpufortrt::internal::check_async_arg(async_arg);
  ::apply_mappings(mappings,
                   num_mappings,
                   gpufortrt_counter_structured,
                   blocking,/*blocking*/
                   async_val,
                   false/*finalize*/);
}

void gpufortrt_data_end_async(int async_arg) {
  if ( !gpufortrt::internal::initialized ) LOG_ERROR("gpufortrt_data_end_async: runtime not initialized")
  gpufortrt_queue_t queue = gpufortrt::internal::queue_record_list.use_create_queue(async_arg);
  gpufortrt::internal::structured_region_stack.leave_structured_region(true,queue);
}

void gpufortrt_enter_exit_data(gpufortrt_mapping_t* mappings,
                               int num_mappings,
                               bool finalize) {
  if ( !gpufortrt::internal::initialized ) gpufortrt_init();
  ::apply_mappings(mappings,
                 num_mappings,
                 gpufortrt_counter_dynamic,
                 true,gpufortrt_async_noval,
                 finalize);
}

void gpufortrt_enter_exit_data_async(gpufortrt_mapping_t* mappings,
                                     int num_mappings,
                                     int async_arg,
                                     bool finalize) {
  if ( !gpufortrt::internal::initialized ) gpufortrt_init();
  bool blocking; int async_val;
  std::tie(blocking,async_val) = gpufortrt::internal::check_async_arg(async_arg);
  ::apply_mappings(mappings,
                 num_mappings,
                 gpufortrt_counter_dynamic,
                 blocking,/*blocking*/
                 async_val,
                 finalize);
}

void* gpufortrt_present(void* hostptr,std::size_t num_bytes) {
  return ::create_increment_action(
    gpufortrt_counter_dynamic,
    hostptr,
    num_bytes,
    gpufortrt_map_kind_present,
    false,/*never_deallocate*/
    true,/*blocking*/
    gpufortrt_async_noval);
}

void* gpufortrt_create(void* hostptr,std::size_t num_bytes,bool never_deallocate) {
  return ::create_increment_action(
    gpufortrt_counter_dynamic,
    hostptr,
    num_bytes,
    gpufortrt_map_kind_create,
    false,/*never_deallocate*/
    false,/*blocking*/
    gpufortrt_async_noval);
}

void gpufortrt_create_async(void* hostptr,std::size_t num_bytes,int async_arg,bool never_deallocate) {
  bool blocking; int async_val;
  std::tie(blocking,async_val) = gpufortrt::internal::check_async_arg(async_arg);
  // gpufortrt_create(hostptr,num_bytes,never_deallocate);
  ::create_increment_action(
    gpufortrt_counter_dynamic,
    hostptr,
    num_bytes,
    gpufortrt_map_kind_create,
    never_deallocate,/*never_deallocate*/
    blocking,/*blocking*/
    async_val);
}

void gpufortrt_delete(void* hostptr,std::size_t num_bytes) {
  ::decrement_release_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_delete,
    false,/*finalize,*/
    true,/*blocking*/
    gpufortrt_async_noval);
}
void gpufortrt_delete_finalize(void* hostptr,std::size_t num_bytes) {
  ::decrement_release_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_delete,
    true/*finalize*/,
    true,/*blocking*/
    gpufortrt_async_noval);
}
void gpufortrt_delete_async(void* hostptr,std::size_t num_bytes,int async_arg) {
  bool blocking; int async_val;
  std::tie(blocking,async_val) = gpufortrt::internal::check_async_arg(async_arg);
  ::decrement_release_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_delete,
    false/*finalize*/,
    blocking,
    async_val);
}
void gpufortrt_delete_finalize_async(void* hostptr,std::size_t num_bytes,int async_arg) {
  bool blocking; int async_val;
  std::tie(blocking,async_val) = gpufortrt::internal::check_async_arg(async_arg);
  ::decrement_release_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_delete,
    true/*finalize*/,
    blocking,
    async_val);
}

void gpufortrt_copyout(void* hostptr,std::size_t num_bytes) {
  ::decrement_release_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_copyout,
    false/*finalize*/,
    true,/*blocking*/
    gpufortrt_async_noval);
}
void gpufortrt_copyout_finalize(void* hostptr,std::size_t num_bytes) {
  ::decrement_release_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_copyout,
    true/*finalize*/,
    true,/*blocking*/
    gpufortrt_async_noval);
}
void gpufortrt_copyout_async(void* hostptr,std::size_t num_bytes,int async_arg) {
  bool blocking; int async_val;
  std::tie(blocking,async_val) = gpufortrt::internal::check_async_arg(async_arg);
  ::decrement_release_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_copyout,
    false/*finalize*/,
    blocking,
    async_val);
}
void gpufortrt_copyout_finalize_async(void* hostptr,std::size_t num_bytes,int async_arg) {
  bool blocking; int async_val;
  std::tie(blocking,async_val) = gpufortrt::internal::check_async_arg(async_arg);
  ::decrement_release_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_copyout,
    true/*finalize*/,
    blocking,
    async_val);
}

void* gpufortrt_copyin(void* hostptr,std::size_t num_bytes,bool never_deallocate) {
  return ::create_increment_action(
    gpufortrt_counter_dynamic,
    hostptr,
    num_bytes,
    gpufortrt_map_kind_copyin,
    never_deallocate,
    true,/*blocking*/
    gpufortrt_async_noval);
}
void gpufortrt_copyin_async(void* hostptr,std::size_t num_bytes,int async_arg,bool never_deallocate) {
  bool blocking; int async_val;
  std::tie(blocking,async_val) = gpufortrt::internal::check_async_arg(async_arg);
  create_increment_action(
    gpufortrt_counter_dynamic,
    hostptr,
    num_bytes,
    gpufortrt_map_kind_copyin,
    never_deallocate,
    blocking,
    async_val);
}

namespace {
  template <bool update_host,bool blocking>
  void update(
      void* hostptr,
      int num_bytes,
      bool if_arg,
      bool if_present_arg,
      int async_arg) {
    LOG_INFO(1,((update_host) ? "update host" : "update device")
            << ((!blocking) ? " async" : "")
            << "; hostptr:"<<hostptr 
            << ", num_bytes:"<<num_bytes 
            << ", if_arg:"<<if_arg 
            << ", if_present_arg:"<<if_present_arg
            << ((!blocking) ? ", async_arg:" : "")
            << ((!blocking) ? std::to_string(async_arg).c_str() : ""))
    if ( if_arg ) {
      if ( !gpufortrt::internal::initialized ) LOG_ERROR("update: runtime not initialized")
      if ( hostptr != nullptr ) { // nullptr means no-op
        auto list_tuple/*success,loc,offset*/ = gpufortrt::internal::record_list.find_record(hostptr,num_bytes);
        const bool& success = std::get<0>(list_tuple);
        const std::size_t& loc = std::get<1>(list_tuple);
        if ( !success && !if_present_arg ) { 
          LOG_ERROR("update: no record found for hostptr="<<hostptr)
        } else if ( success ) {
          auto& record = gpufortrt::internal::record_list.records[loc];
          gpufortrt_queue_t queue = gpufortrt_default_queue; 
          if ( !blocking ) {
            queue = gpufortrt::internal::queue_record_list.use_create_queue(async_arg);
          }
          if ( update_host ) {
            record.copy_section_to_host(hostptr,num_bytes,blocking,queue);
          } else {
            record.copy_section_to_device(hostptr,num_bytes,blocking,queue);
          }
        }
      }
    }
  }
} // namespace

void gpufortrt_update_self(
    void* hostptr,
    std::size_t num_bytes,
    bool if_arg,
    bool if_present_arg) {
  // update_host,blocking
  ::update<true,true>(hostptr,num_bytes,if_arg,if_present_arg,-1); 
}
void gpufortrt_update_self_async(
    void* hostptr,
    std::size_t num_bytes,
    bool if_arg,
    bool if_present_arg,
    int async_arg) {
  ::update<true,false>(hostptr,num_bytes,if_arg,if_present_arg,async_arg); 
}

void gpufortrt_update_device(
    void* hostptr,
    std::size_t num_bytes,
    bool if_arg,
    bool if_present_arg) {
  ::update<false,true>(hostptr,num_bytes,if_arg,if_present_arg,-1); 
}
void gpufortrt_update_device_async(
    void* hostptr,
    std::size_t num_bytes,
    bool if_arg,
    bool if_present_arg,
    int async_arg) {
  ::update<false,false>(hostptr,num_bytes,if_arg,if_present_arg,async_arg); 
}

void gpufortrt_wait_all(bool if_arg) {
  if ( if_arg ) {
    HIP_CHECK(hipDeviceSynchronize()) // TODO backend specific, externalize
  }
}
void gpufortrt_wait(int* wait_arg,
                    int num_wait,
                    bool if_arg) {
  if ( if_arg ) {
    for (int i = 0; i < num_wait; i++) {
      auto queue = gpufortrt::internal::queue_record_list.use_create_queue(wait_arg[i]);
      HIP_CHECK(hipStreamSynchronize(queue)) // TODO backend specific, externalize
    }
  }
}
void gpufortrt_wait_async(int* wait_arg,int num_wait,
                          int* async_arg,int num_async,
                          bool if_arg) {
  if ( if_arg ) {
    for (int i = 0; i < num_wait; i++) {
      hipEvent_t event;// TODO backend specific, externalize
      HIP_CHECK(hipEventCreateWithFlags(&event,hipEventDisableTiming))// TODO backend specific, externalize
      auto queue = gpufortrt::internal::queue_record_list.use_create_queue(wait_arg[i]);
      HIP_CHECK(hipEventRecord(event,queue))// TODO backend specific, externalize
      for (int j = 0; j < num_async; j++) {
        auto queue_async = gpufortrt::internal::queue_record_list.use_create_queue(async_arg[j]);
        HIP_CHECK(hipStreamWaitEvent(queue_async,event,0)) // TODO backend specific, externalize
      }
    }
  }
}
void gpufortrt_wait_all_async(int* async_arg,int num_async,
                              bool if_arg) {
  if ( if_arg ) {
    hipEvent_t event;// TODO backend specific, externalize
    HIP_CHECK(hipEventCreateWithFlags(&event,hipEventDisableTiming))// TODO backend specific, externalize
    HIP_CHECK(hipEventRecord(event,gpufortrt_default_queue))// TODO backend specific, externalize
    for (int j = 0; j < num_async; j++) {
      auto queue_async = gpufortrt::internal::queue_record_list.use_create_queue(async_arg[j]);
      HIP_CHECK(hipStreamWaitEvent(queue_async,event,0)) // TODO backend specific, externalize
    }
  }
}

void gpufortrt_wait_device(int* wait_arg, int num_wait, int dev_num, bool if_arg){
  const int current_device_num = gpufortrt_get_device_num();
  gpufortrt_set_device_num(dev_num);
  gpufortrt_wait(wait_arg, num_wait, if_arg);
  gpufortrt_set_device_num(current_device_num);
}

void gpufortrt_wait_device_async(int* wait_arg, int num_wait, 
                                 int* async_arg, int num_async, 
                                 int dev_num, bool if_arg){
  const int current_device_num = gpufortrt_get_device_num();
  gpufortrt_set_device_num(dev_num);
  gpufortrt_wait_async(wait_arg, num_wait, async_arg, num_async, if_arg);
  gpufortrt_set_device_num(current_device_num);
}

void gpufortrt_wait_all_device(int dev_num, bool if_arg){
  const int current_device_num = gpufortrt_get_device_num();
  gpufortrt_set_device_num(dev_num);
  gpufortrt_wait_all(if_arg);
  gpufortrt_set_device_num(current_device_num);
}

void gpufortrt_wait_all_device_async(int* async_arg, int num_async, 
                                 int dev_num, bool if_arg){
  const int current_device_num = gpufortrt_get_device_num();
  gpufortrt_set_device_num(dev_num);
  gpufortrt_wait_all_async(async_arg, num_async, if_arg);
  gpufortrt_set_device_num(current_device_num);
}

int gpufortrt_async_test(int wait_arg) {
  return gpufortrt::internal::queue_record_list.test(wait_arg);
}

int gpufortrt_async_test_device(int wait_arg, int dev_num) {
  const int current_device_num = gpufortrt_get_device_num();
  gpufortrt_set_device_num(dev_num);
  int result = gpufortrt_async_test(wait_arg);
  gpufortrt_set_device_num(current_device_num);
  return result;
}

int gpufortrt_async_test_all() {
  for (size_t i = 0; i < gpufortrt::internal::queue_record_list.size(); i++) {
    auto& queue = gpufortrt::internal::queue_record_list[i].queue;
    if(hipStreamQuery(queue) != hipSuccess) return 0;
    // HIP_CHECK(hipStreamQuery(queue))// TODO backend specific, externalize
  } 
  return 1;
}

int gpufortrt_async_test_all_device(int dev_num) {
  const int current_device_num = gpufortrt_get_device_num();
  gpufortrt_set_device_num(dev_num);
  int result = gpufortrt_async_test(dev_num);
  gpufortrt_set_device_num(current_device_num);
  return result;
}
  
void* gpufortrt_deviceptr(void* hostptr) {
  LOG_INFO(1,"deviceptr; "
          << "; hostptr: "<<hostptr )
  if ( hostptr == nullptr ) {
    return nullptr;
  } else {
    auto stack_tuple/*success,&record,offset,use_hostptr*/ = gpufortrt::internal::structured_region_stack.find_record(hostptr);
    bool& success = std::get<0>(stack_tuple);
    auto*& record = std::get<1>(stack_tuple);
    std::ptrdiff_t& offset = std::get<2>(stack_tuple);
    bool& use_hostptr = std::get<3>(stack_tuple);
    //
    if ( !success && !use_hostptr ) {
      auto list_tuple/*success,loc,offset*/ = gpufortrt::internal::record_list.find_record(hostptr);
      success = std::get<0>(list_tuple);
      const std::size_t& loc = std::get<1>(list_tuple);
      offset = std::get<2>(list_tuple);
      //
      if ( success ) {
        record = &gpufortrt::internal::record_list.records[loc];
      }
    }
    // above code may overwrite record
    if ( success ) {
      void* result = static_cast<void*>(static_cast<char*>(record->deviceptr) + offset);
      LOG_INFO(2,"<deviceptr"
               << "; return deviceptr=" << result 
               << "; record_deviceptr:" << record->deviceptr
               << ", offset:" << offset
               << ", use_hostptr:0")
      return result;
    } else if ( use_hostptr ) {
      LOG_INFO(2,"<deviceptr"
               << "; return hostptr=" << hostptr
               << "; use_hostptr:1")
      return hostptr;
    } else {
      LOG_ERROR("<deviceptr: hostptr="<<hostptr<<" not mapped");
      return nullptr; /* terminates beforehand */
    } 
  }
}

void* gpufortrt_use_device(void* hostptr,bool if_arg,bool if_present_arg) {
  if ( !gpufortrt::internal::initialized ) LOG_ERROR("gpufortrt_use_device: runtime not initialized")
  LOG_INFO(1,"use_device"
          << "; hostptr: "<<hostptr 
          << ", if_arg: "<<if_arg 
          << ", if_present_arg: "<<if_present_arg)
  if ( hostptr == nullptr ) {
     LOG_INFO(2,"<use_device; return nullptr; hostptr=nullptr")
     return nullptr;
  } else if ( if_arg ) {
    auto list_tuple/*success,loc,offset*/ = gpufortrt::internal::record_list.find_record(hostptr); 
    const bool& success = std::get<0>(list_tuple);
    const std::size_t& loc = std::get<1>(list_tuple);
    const std::ptrdiff_t& offset = std::get<2>(list_tuple);
    //
    if ( success ) {
      auto& record = gpufortrt::internal::record_list.records[loc];
      LOG_INFO(2,"<use_device; return deviceptr="<<hostptr<<" for hostptr="<<hostptr
               <<" (record_hostptr: "<<record.hostptr<<", offset: "<<offset<<" B)")
      return gpufortrt::internal::offsetted_record_deviceptr(record,offset);
    } else if ( if_present_arg ) {
      LOG_INFO(2,"<use_device; return hostptr="<<hostptr<<"; no record present, if_present_arg: 0")
      return hostptr;
    } else {
      LOG_ERROR("gpufortrt_use_device: no record found for hostptr=" << hostptr << ", if_present_arg: 0")
      return nullptr; /* terminates beforehand */
    }
  } else {
    LOG_INFO(2,"<use_device; return hostptr="<<hostptr<<"; if_arg: 0")
    return hostptr;
  }
}

bool gpufortrt_is_present(void* hostptr,std::size_t num_bytes) {
  if ( !gpufortrt::internal::initialized ) LOG_ERROR("gpufortrt_is_present: runtime not initialized")
  if ( hostptr != nullptr ) { // nullptr means no-op
    auto list_tuple/*success,loc,offset*/ = gpufortrt::internal::record_list.find_record(hostptr,num_bytes);
    return std::get<0>(list_tuple);
  } else{
      return false;
  }
}
