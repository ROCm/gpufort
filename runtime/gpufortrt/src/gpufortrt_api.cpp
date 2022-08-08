// SPDX-License-Identifier: MIT
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "internal/gpufortrt_core.h"

#include <string>
#include <sstream>

#include <hip/hip_runtime_api.h>

#include "gpufortrt_auxiliary.h"

gpufortrt_async_noval = -1;

void gpufortrt_mapping_init(
    gpufortrt_mapping_t* mapping,
    void* hostptr,
    size_t num_bytes,
    gpufortrt_map_kind_t map_kind,
    bool never_deallocate) {
  mapping->hostptr    = hostptr;
  mapping->num_bytes  = num_bytes;
  mapping->map_kind   = map_kind;
  mapping->never_deallocate = never_deallocate;
}

void gpufortrt_init() {
  if ( gpufortrt::internal::initialized ) {
    throw std::invalid_argument("init: runtime has already been initialized");
  else {
    set_from_environment(gpufortrt::internal::LOG_LEVEL,"GPUFORTRT_LOG_LEVEL");
    set_from_environment(gpufortrt::internal::INITIAL_RECORDS_CAPACITY,"GPUFORTRT_INITIAL_RECORDS_CAPACITY");
    set_from_environment(gpufortrt::internal::INITIAL_QUEUE_RECORDS_CAPACITY,"GPUFORTRT_INITIAL_QUEUE_RECORDS_CAPACITY");
    set_from_environment(gpufortrt::internal::INITIAL_STRUCTURED_REGION_STACK_CAPACITY,"GPUFORTRT_INITIAL_STRUCTURED_REGION_STACK_CAPACITY");
    set_from_environment(gpufortrt::internal::BLOCK_SIZE,"GPUFORTRT_BLOCK_SIZE");
    set_from_environment(gpufortrt::internal::REUSE_THRESHOLD,"GPUFORTRT_REUSE_THRESHOLD");
    set_from_environment(gpufortrt::internal::NUM_REFS_TO_DEALLOCATE,"GPUFORTRT_NUM_REFS_TO_DEALLOCATE");
    gpufortrt::internal::record_list.reserve(gpufortrt::internal::INITIAL_RECORDS_CAPACITY);
    gpufortrt::internal::record_queue_list.reserve(gpufortrt::internal::INITIAL_QUEUE_RECORDS_CAPACITY);
    gpufortrt::internal::structured_region_stack.reserve(gpufortrt::internal::INITIAL_STRUCTURED_REGION_STACK_CAPACITY)
    gpufortrt::internal::initialized = true;
    //
    LOG_INFO(1,"initialized runtime")
    LOG_INFO(1,"GPUFORTRT_LOG_LEVEL=" << gpufortrt::internal::LOG_LEVEL)
    LOG_INFO(1,"GPUFORTRT_INITIAL_RECORDS_CAPACITY=" << gpufortrt::internal::INITIAL_RECORDS_CAPACITY)
    LOG_INFO(1,"GPUFORTRT_INITIAL_QUEUE_RECORDS_CAPACITY=" << gpufortrt::internal::INITIAL_QUEUE_RECORDS_CAPACITY)
    LOG_INFO(1,"GPUFORTRT_INITIAL_STRUCTURED_REGION_STACK_CAPACITY=" << gpufortrt::internal::INITIAL_STRUCTURED_REGION_STACK_CAPACITY)
    LOG_INFO(1,"GPUFORTRT_BLOCK_SIZE=" << gpufortrt::internal::BLOCK_SIZE)
    LOG_INFO(1,"GPUFORTRT_REUSE_THRESHOLD=" << gpufortrt::internal::REUSE_THRESHOLD)
    LOG_INFO(1,"GPUFORTRT_NUM_REFS_TO_DEALLOCATE=" << gpufortrt::internal::NUM_REFS_TO_DEALLOCATE)
  }
}

void gpufortrt_shutdown() {
  if ( !gpufortrt::internal::initialized ) {
    LOG_ERROR("gpufortrt_shutdown: runtime has not been initialized")
  }
  gpufortrt::record_list.destroy();
  gpufortrt::internal::queue_record_list.destroy();
  LOG_INFO(1,"terminated runtime")
}

namespace {
  void* run_use_increment_action(
      void* hostptr,
      size_t num_bytes,
      gpufortrt_map_kind_t map_kind,
      bool blocking,
      int async,
      bool never_deallocate,
      gpufortrt_counter_t ctr_to_update) {
    if ( ! initialized_ ) gpufortrt_init();
    if ( hostptr == nullptr) {
      #ifndef NULLPTR_MEANS_NOOP
      std::stringstream ss;
      ss << "gpufortrt_" << map_kind << ": " << "hostptr must not be null"; 
      throw std::invalid_argument(ss.str());
      #endif
      return nullptr;
    } else {
      gpufortrt_queue_t queue = gpufortrt_default_queue;
      if ( !blocking ) {
        queue = gpufortrt::internal::queue_record_list.use_create_queue(async);
      }
      size_t loc = gpufortrt::internal::record_list.use_increment_record(
        hostptr,
        num_bytes,
        map_kind,
        ctr_to_update,
        blocking,
        queue,
        never_deallocate);

      auto record = gpufortrt::internal::record_list[loc];
      if ( ctr_to_update == gpufortrt_counter_structured ) {
        gpufortrt::internal::structured_region_stack.push_back(record);
      }
      return record.deviceptr;
    }
  }

  void apply_mappings(gpufortrt_mapping_t* mappings,
                      int num_mappings,
                      gpufortrt_counter_t ctr_to_update,
                      bool blocking,
                      int async,bool finalize) {
    for (int i = 0; i < num_mappings; i++) {
       auto mapping = mappings[i];
       switch (mapping.map_kind) {
         case gpufortrt_map_kind_delete:
            if ( blocking ) {
              if ( finalize ) {
                gpufortrt_delete_finalize(
                  mapping.hostptr,
                  ctr_to_update);
              } else {
                gpufortrt_delete(
                  mapping.hostptr,
                  ctr_to_update);
              }
            } else {
              if ( finalize ) {
                gpufortrt_delete_finalize_async(
                  mapping.hostptr,
                  async,
                  ctr_to_update);
              } else {
                gpufortrt_delete_async(
                  mapping.hostptr,
                  async,
                  ctr_to_update);
              }
            }
            break;
         case gpufortrt_map_kind_present:
         case gpufortrt_map_kind_create:
         case gpufortrt_map_kind_no_create:
         case gpufortrt_map_kind_copyin:
         case gpufortrt_map_kind_copyout:
         case gpufortrt_map_kind_copy:
            ::run_use_increment_action(
              mapping.hostptr,
              mapping.num_bytes,
              mapping.map_kind
              blocking,
              async,
              mapping.never_deallocate, 
              ctr_to_update);
            break;
         default: std::invalid_argument("apply_mappings: invalid map_kind"); break;
       }
    }
  }

}

void gpufortrt_data_start(gpufortrt_mapping_t* mappings,int num_mappings) {
  gpufortrt::internal::structured_region_stack.enter_structured_region();
  ::apply_mappings(mappings,
                   num_mappings,
                   gpufortrt_counter_structured,
                   true,gpufortrt_async_noval,false/*finalize*/);
}

void gpufortrt_data_end() {
  gpufortrt::internal::structured_region_stack.leave_structured_region();
}

void gpufortrt_data_start_async(gpufortrt_mapping_t* mappings,int num_mappings,int async) {
  gpufortrt::internal::structured_region_stack.enter_structured_region();
  ::apply_mappings(mappings,
                   num_mappings,
                   gpufortrt_counter_structured,
                   false,async,false/*finalize*/);
}

void gpufortrt_data_end_async(int async) {
  gpufortrt_queue_t queue = gpufortrt::internal::queue_record_list.use_create_queue(async);
  gpufortrt::internal::structured_region_stack.leave_structured_region(true,queue);
}

void gpufortrt_enter_exit_data(gpufortrt_mapping_t* mappings,
                               int num_mappings,
                               bool finalize) {
  ::apply_mappings(mappings,
                 num_mappings,
                 gpufortrt_counter_dynamic,
                 true,gpufortrt_async_noval,
                 finalize);
}

void gpufortrt_enter_exit_data_async(gpufortrt_mapping_t* mappings,
                                     int num_mappings,
                                     int async,
                                     bool finalize) {
  ::apply_mappings(mappings,
                 num_mappings,
                 gpufortrt_counter_dynamic,
                 false,async,
                 finalize);
}

void* gpufortrt_use_device(void* hostptr,size_t num_bytes,
                              bool condition,bool if_present) {
  void* resultptr = hostptr;
  if ( hostptr == nullptr ) {
     return nullptr;
  } else if ( condition ) {
    size_t loc = record_list_find_record_(hostptr,success);
    // TODO already detect a suitable candidate in find_record on-the-fly
    if ( success ) {
      auto& record = gpufortrt::internal::record_list.records[loc];
      size_t offset_bytes;
      bool fits = record.is_subarray(hostptr,num_bytes,offset_bytes/*inout*/); // TODO might not fit, i.e. only subarray
                                                                      // might have been mapped before
      if ( !fits ) LOG_ERROR("gpufortrt_use_device: input data region overlaps with previously mapped data but is no subset of it") 
      return static_cast<void*>(static_cast<char*>(record.deviceptr) + offset_bytes);
    } else if ( if_present ) {
      return hostptr;
    } else {
      LOG_ERROR("gpufortrt_use_device: did not find record for hostptr " << hostptr)
    }
  }
}

void gpufortrt_delete(
  void* hostptr,
  gpufortrt_counter_t ctr_to_update) {
  if ( !initialized_ ) LOG_ERROR("gpufortrt_delete: runtime not initialized")
  if ( hostptr == nullptr ) {
#ifndef NULLPTR_MEANS_NOOP
    throw std::invalid_argument("gpufortrt_delete: hostptr is nullptr");
#endif
    return
  else {
    gpufortrt::internal::record_list.decrement_release_record(
      hostptr,
      gpufortrt_counter_dynamic,
      veto_copy_to_host,
      blocking,
      queue,
      .false.);
  }
}

void gpufortrt_delete_finalize(void* hostptr) {
  if ( !initialized_ ) LOG_ERROR("gpufortrt_delete: runtime not initialized")
  if ( hostptr == nullptr ) {
#ifndef NULLPTR_MEANS_NOOP
    throw std::invalid_argument("gpufortrt_delete: hostptr is nullptr");
#endif
    return
  else {
    gpufortrt::internal::record_list.decrement_release_record(
      hostptr,
      gpufortrt_counter_dynamic,
      veto_copy_to_host,
      blocking,
      queue,
      .true.);
  }
}

void* gpufortrt_present(void* hostptr,size_t num_bytes,
                        gpufortrt_counter_t ctr_to_update) {
  return ::run_use_increment_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_present,
    async,
    false, 
    ctr_to_update);
}

void* gpufortrt_no_create(void* hostptr,size_t num_bytes,bool never_deallocate,
                            gpufortrt_counter_t ctr_to_update) {
  return ::run_use_increment_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_no_create,
    async,
    never_deallocate, 
    ctr_to_update);
}



void* gpufortrt_create(void* hostptr,size_t num_bytes,bool never_deallocate,
                         gpufortrt_counter_t ctr_to_update) {
  return ::run_use_increment_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_copyin,
    async,
    never_deallocate, 
    ctr_to_update);
}

void* gpufortrt_copyin(void* hostptr,size_t num_bytes,int async,bool never_deallocate,
                          gpufortrt_counter_t ctr_to_update) {
  return ::run_use_increment_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_copyin,
    async,
    never_deallocate, 
    ctr_to_update);
}

void* gpufortrt_copyout(void* hostptr,size_t num_bytes,int async,bool never_deallocate,
                           gpufortrt_counter_t ctr_to_update) {
  return ::run_use_increment_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_copyout,
    async,
    never_deallocate,
    ctr_to_update);
}

void* gpufortrt_copy(void* hostptr,size_t num_bytes,bool never_deallocate,
                     gpufortrt_counter_t ctr_to_update) {
  return ::run_use_increment_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_copy,
    async,
    never_deallocate,
    ctr_to_update);
}

void* gpufortrt_copy_async(void* hostptr,size_t num_bytes,int async,bool never_deallocate,
                     gpufortrt_counter_t ctr_to_update) {
  return ::run_use_increment_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_copy,
    async,
    never_deallocate,
    ctr_to_update);
}

namespace {
  template <bool update_host,bool update_section,bool blocking>
  void update(
      void* hostptr,
      int num_bytes,
      bool condition,
      bool if_present,
      int async_arg) {
    if ( condition ) {
      if ( !initialized_ ) LOG_ERROR("update_host: runtime not initialized")
      if ( hostptr == nullptr ) {
        #if NULLPTR_MEANS_NOOP
        throw std::invalid_argument("update_host: hostptr is nullptr")
        #endif
        return;
      }
      size_t loc = gpufortrt::internal::record_list.find_record(hostptr,success);
      if ( !success && !if_present ) { 
        LOG_ERROR("update_host: no deviceptr found for hostptr")
      } else if ( success ) {
        auto record = gpufortrt::internal::record_list.records[loc];
        gpufortrt_queue_t queue = gpufortrt_default_queue; 
        if ( !blocking ) {
          queue = gpufortrt::internal::queue_record_list.use_create_queue(async);
        }
        if ( update_host ) {
          if ( update_section ) {
            record.copy_section_to_host(hostptr,num_bytes,blocking,queue);
          } else {
            record.copy_to_host(blocking,queue);
          }
        } else {
          if ( update_section ) {
            record.copy_section_to_device(hostptr,num_bytes,blocking,queue);
          } else {
            record.copy_to_device(blocking,queue);
          }
        }
      }
    }
  }
}

void gpufortrt_update_host(
    void* hostptr,
    bool condition,
    bool if_present) {
  ::update<true,false,true>(hostptr,0,condition,if_present,-1); 
}
void gpufortrt_update_self_async(
    void* hostptr,
    bool condition,
    bool if_present,
    int async) {
  ::update<false,false,true>(hostptr,0,condition,if_present,async); 
}
void gpufortrt_update_self_section(
    void* hostptr,
    size_t num_bytes,
    bool condition,
    bool if_present) {
  ::update<true,true,true>(hostptr,num_bytes,condition,if_present,-1); 
}
void gpufortrt_update_self_section_async(
    void* hostptr,
    size_t num_bytes,
    bool condition,
    bool if_present,
    int async) {
  ::update<false,true,true>(hostptr,num_bytes,condition,if_present,async); 
}

void gpufortrt_update_device(
    void* hostptr,
    bool condition,
    bool if_present) {
  ::update<true,false,false>(hostptr,0,condition,if_present,-1); 
}
void gpufortrt_update_device_async(
    void* hostptr,
    bool condition,
    bool if_present,
    int async) {
  ::update<false,false,false>(hostptr,0,condition,if_present,async); 
}
void gpufortrt_update_device_section(
    void* hostptr,
    size_t num_bytes,
    bool condition,
    bool if_present) {
  ::update<true,false,false>(hostptr,num_bytes,condition,if_present,-1); 
}
void gpufortrt_update_device_section_async(
    void* hostptr,
    size_t num_bytes,
    bool condition,
    bool if_present,
    int async) {
  ::update<false,false,false>(hostptr,num_bytes,condition,if_present,async); 
}

void gpufortrt_wait_all(bool condition) {
  if ( condition ) {
    HIP_CHECK(hipDeviceSynchronize()) // TODO backend specific, externalize
  }
}
void gpufortrt_wait(int* wait_arg,
                    int num_wait,
                    bool condition) {
  if ( condition ) {
    for (int i = 0; i < num_wait; i++) {
      auto queue_record = gpufortrt::internal::queue_record_list.use_create_queue(wait_arg[i]]);
      HIP_CHECK(hipStreamSynchronize(queue_record.queue)) // TODO backend specific, externalize
    }
  }
}
void gpufortrt_wait_async(int* wait_arg,int num_wait,
                          int* async,int num_async,
                          bool condition) {
  if ( condition ) {
    for (int i = 0; i < num_wait; i++) {
      hipEvent_t event;// TODO backend specific, externalize
      HIP_CHECK(hipEventCreateWithFlags(event,hipEventDisableTiming))// TODO backend specific, externalize
      auto queue_record = gpufortrt::internal::queue_record_list.use_create_queue(wait_arg[i]]);
      HIP_CHECK(hipEventRecord(event,record.queue))// TODO backend specific, externalize
      for (int j = 0; j < num_async; j++) {
        auto queue_record_async = gpufortrt::internal::queue_record_list.use_create_queue(async[i]]);
        HIP_CHECK(hipStreamWaitEvent(queue_record_async.queue,event);// TODO backend specific, externalize
      }
    }
  }
}
void gpufortrt_wait_all_async(int* async,int num_async,
                              bool condition) {
  if ( condition ) {
    for (int i = 0; i < num_wait; i++) {
      hipEvent_t event;// TODO backend specific, externalize
      HIP_CHECK(hipEventCreateWithFlags(event,hipEventDisableTiming))// TODO backend specific, externalize
      HIP_CHECK(hipEventRecord(event,nullptr))// TODO backend specific, externalize
      for (int j = 0; j < num_async; j++) {
        auto queue_record_async = gpufortrt::internal::queue_record_list.use_create_queue(async[i]]);
        HIP_CHECK(hipStreamWaitEvent(queue_record_async.queue,event);// TODO backend specific, externalize
      }
    }
  }
}
  
void* gpufortrt_deviceptr(void* hostptr) {
  auto* record = gpufortrt::internal::structured_region_stack.find_in_current_region(hostptr);
  if ( record == nullptr ) {
    bool success = false;
    size_t loc = record_list_find_record_(hostptr,success);
    if ( success ) {
      record = &gpufortrt::internal::record_list.records[loc];
    }
  }
  if ( record != nullptr ) {
    size_t offset_bytes;
    bool fits = record->is_subarray(hostptr,0,offset_bytes/*inout*/);
    return static_cast<void*>(static_cast<char*>(record->deviceptr) + offset_bytes);
  } else {
    return nullptr
  } 
}
