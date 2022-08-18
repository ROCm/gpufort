// SPDX-License-Identifier: MIT
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "gpufortrt_api.h"

#include <string>
#include <sstream>

#include "hip/hip_runtime_api.h"

#include "internal/auxiliary.h"
#include "internal/gpufortrt_core.h"

int gpufortrt_async_noval = -1;

void gpufortrt_mapping_init(
    gpufortrt_mapping_t* mapping,
    void* hostptr,
    size_t num_bytes,
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
    LOG_INFO(1,"  GPUFORTRT_LOG_LEVEL=" << gpufortrt::internal::LOG_LEVEL)
    LOG_INFO(1,"  GPUFORTRT_INITIAL_RECORDS_CAPACITY=" << gpufortrt::internal::INITIAL_RECORDS_CAPACITY)
    LOG_INFO(1,"  GPUFORTRT_INITIAL_QUEUE_RECORDS_CAPACITY=" << gpufortrt::internal::INITIAL_QUEUE_RECORDS_CAPACITY)
    LOG_INFO(1,"  GPUFORTRT_INITIAL_STRUCTURED_REGION_STACK_CAPACITY=" << gpufortrt::internal::INITIAL_STRUCTURED_REGION_STACK_CAPACITY)
    LOG_INFO(1,"  GPUFORTRT_BLOCK_SIZE=" << gpufortrt::internal::BLOCK_SIZE)
    LOG_INFO(1,"  GPUFORTRT_REUSE_THRESHOLD=" << gpufortrt::internal::REUSE_THRESHOLD)
    LOG_INFO(1,"  GPUFORTRT_NUM_REFS_TO_DEALLOCATE=" << gpufortrt::internal::NUM_REFS_TO_DEALLOCATE)
  }
}

void gpufortrt_shutdown() {
  LOG_INFO(1,"shutdown;")
  if ( !gpufortrt::internal::initialized ) {
    LOG_ERROR("gpufortrt_shutdown: runtime has not been initialized")
  }
  gpufortrt::internal::record_list.destroy();
  gpufortrt::internal::queue_record_list.destroy();
}

namespace {
  void* run_use_increment_action(
      void* hostptr,
      size_t num_bytes,
      gpufortrt_map_kind_t map_kind,
      bool blocking,
      int async_arg,
      bool never_deallocate,
      gpufortrt_counter_t ctr_to_update) {
    if ( ! gpufortrt::internal::initialized ) gpufortrt_init();
    LOG_INFO(2, map_kind
             << ((!blocking) ? "_async" : "")
             << "; hostptr:" << hostptr 
             << ", num_bytes:" << num_bytes 
             << ((!blocking) ? ", async_arg:" : "")
             << ((!blocking) ? std::to_string(async_arg).c_str() : "")
             << ", never_deallocate:" << never_deallocate 
             << ", ctr_to_update:" << ctr_to_update)
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
        queue = gpufortrt::internal::queue_record_list.use_create_queue(async_arg);
      }
      size_t loc = gpufortrt::internal::record_list.use_increment_record(
        hostptr,
        num_bytes,
        map_kind,
        ctr_to_update,
        blocking,
        queue,
        never_deallocate);

      auto& record = gpufortrt::internal::record_list[loc];
      if ( ctr_to_update == gpufortrt_counter_structured ) {
        gpufortrt::internal::structured_region_stack.push_back(record);
      }
      return record.deviceptr;
    }
  }
  
  void run_decrement_release_action(void* hostptr,size_t num_bytes,int async_arg,
                                    bool blocking, bool finalize, bool copyout) {
    LOG_INFO(2, ((copyout) ? "copyout" : "delete")
            << ((finalize) ? "_finalize" : "")
            << ((!blocking) ? "_async" : "")
            << "; hostptr:"<<hostptr 
            << ((!blocking) ? ", async_arg:" : "")
            << ((!blocking) ? std::to_string(async_arg).c_str() : ""))
    if ( !gpufortrt::internal::initialized ) LOG_ERROR("gpufortrt_delete: runtime not initialized")
    if ( hostptr == nullptr ) {
  #ifndef NULLPTR_MEANS_NOOP
      throw std::invalid_argument("hostptr is nullptr");
  #endif
    } else {
      gpufortrt_queue_t queue = gpufortrt_default_queue;
      if ( !blocking ) {
        queue = gpufortrt::internal::queue_record_list.use_create_queue(async_arg);
      }
      gpufortrt::internal::record_list.unstructured_decrement_release_record(
        hostptr,
        num_bytes,
        copyout,
        finalize,
        blocking,
        queue);
    }
  }

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
           ::run_decrement_release_action(
               mapping.hostptr,
               mapping.num_bytes,
               async_arg,
               blocking,
               finalize,
               false/*copyout*/);
           break;
        case gpufortrt_map_kind_copyout:
           switch ( ctr_to_update ) {
             case gpufortrt_counter_structured:
               ::run_use_increment_action(
                 mapping.hostptr,
                 mapping.num_bytes,
                 mapping.map_kind,
                 blocking,
                 async_arg,
                 mapping.never_deallocate, 
                 ctr_to_update);
               break;
             case gpufortrt_counter_dynamic:
               ::run_decrement_release_action(
                   mapping.hostptr,
                   mapping.num_bytes,
                   async_arg,
                   blocking,
                   finalize,
                   true/*copyout*/);
               break;
             default: std::invalid_argument("apply_mappings: invalid ctr_to_update"); break;
           }
           break;
        case gpufortrt_map_kind_present:
        case gpufortrt_map_kind_create:
        case gpufortrt_map_kind_no_create:
        case gpufortrt_map_kind_copyin:
        case gpufortrt_map_kind_copy:
           ::run_use_increment_action(
             mapping.hostptr,
             mapping.num_bytes,
             mapping.map_kind,
             blocking,
             async_arg,
             mapping.never_deallocate, 
             ctr_to_update);
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
  LOG_INFO(1,"data end") 
  if ( !gpufortrt::internal::initialized ) LOG_ERROR("gpufortrt_data_end: runtime not initialized")
  gpufortrt::internal::structured_region_stack.leave_structured_region(false,nullptr);
}

void gpufortrt_data_start_async(gpufortrt_mapping_t* mappings,int num_mappings,int async_arg) {
  if ( !gpufortrt::internal::initialized ) gpufortrt_init();
  gpufortrt::internal::structured_region_stack.enter_structured_region();
  ::apply_mappings(mappings,
                   num_mappings,
                   gpufortrt_counter_structured,
                   false,async_arg,false/*finalize*/);
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
  ::apply_mappings(mappings,
                 num_mappings,
                 gpufortrt_counter_dynamic,
                 false,async_arg,
                 finalize);
}

void* gpufortrt_use_device(void* hostptr,bool condition,bool if_present) {
  if ( !gpufortrt::internal::initialized ) LOG_ERROR("gpufortrt_use_device: runtime not initialized")
  void* resultptr = hostptr;
  if ( hostptr == nullptr ) {
     return nullptr;
  } else if ( condition ) {
    bool success = false;
    size_t loc = gpufortrt::internal::record_list.find_record(hostptr,1/*num_bytes*/,success/*inout*/); 
        // success: hostptr in [record.hostptr,record.hostptr+record.num_bytes)
    if ( success ) {
      auto& record = gpufortrt::internal::record_list.records[loc];
      size_t offset_bytes;
      record.is_host_data_subset(hostptr,1/*num_bytes*/,offset_bytes/*inout*/); // get offset_bytes
      return static_cast<void*>(static_cast<char*>(record.deviceptr) + offset_bytes);
    } else if ( if_present ) {
      return hostptr;
    } else {
      LOG_ERROR("gpufortrt_use_device: did not find record for hostptr " << hostptr)
    }
  }
  return nullptr; 
}

void* gpufortrt_present(void* hostptr,size_t num_bytes) {
  return ::run_use_increment_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_present,
    false,
    gpufortrt_async_noval,
    false, 
    gpufortrt_counter_dynamic);
}

void* gpufortrt_create(void* hostptr,size_t num_bytes,bool never_deallocate) {
  return ::run_use_increment_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_copyin,
    false,
    gpufortrt_async_noval,
    never_deallocate, 
    gpufortrt_counter_dynamic);
}

void* gpufortrt_create_async(void* hostptr,size_t num_bytes,int async_arg,bool never_deallocate) {
  return gpufortrt_create(hostptr,num_bytes,never_deallocate);
}

void* gpufortrt_no_create(void* hostptr,size_t num_bytes,bool never_deallocate) {
  return gpufortrt_create(hostptr,num_bytes,never_deallocate);
}

void* gpufortrt_no_create_async(void* hostptr,size_t num_bytes,int async_arg,bool never_deallocate) {
  return gpufortrt_create(hostptr,num_bytes,never_deallocate);
}

void gpufortrt_delete(void* hostptr,size_t num_bytes) {
  ::run_decrement_release_action(hostptr,num_bytes,gpufortrt_async_noval,true,false,false);
}
void gpufortrt_delete_finalize(void* hostptr,size_t num_bytes) {
  ::run_decrement_release_action(hostptr,num_bytes,gpufortrt_async_noval,true,true,false);
}
void gpufortrt_delete_async(void* hostptr,size_t num_bytes,int async_arg) {
  ::run_decrement_release_action(hostptr,num_bytes,async_arg,false,false,false);
}
void gpufortrt_delete_finalize_async(void* hostptr,size_t num_bytes,int async_arg) {
  ::run_decrement_release_action(hostptr,num_bytes,async_arg,false,true,false);
}

void gpufortrt_copyout(void* hostptr,size_t num_bytes) {
  ::run_decrement_release_action(hostptr,num_bytes,gpufortrt_async_noval,true,false,true);
}
void gpufortrt_copyout_finalize(void* hostptr,size_t num_bytes) {
  ::run_decrement_release_action(hostptr,num_bytes,gpufortrt_async_noval,true,true,true);
}
void gpufortrt_copyout_async(void* hostptr,size_t num_bytes,int async_arg) {
  ::run_decrement_release_action(hostptr,num_bytes,async_arg,false,false,true);
}
void gpufortrt_copyout_finalize_async(void* hostptr,size_t num_bytes,int async_arg) {
  ::run_decrement_release_action(hostptr,num_bytes,async_arg,false,true,true);
}

void* gpufortrt_copyin(void* hostptr,size_t num_bytes,bool never_deallocate) {
  return ::run_use_increment_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_copyin,
    true,
    gpufortrt_async_noval,
    never_deallocate, 
    gpufortrt_counter_dynamic);
}
void* gpufortrt_copyin_async(void* hostptr,size_t num_bytes,int async_arg,bool never_deallocate) {
  return ::run_use_increment_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_copyin,
    false,
    async_arg,
    never_deallocate, 
    gpufortrt_counter_dynamic);
}

void* gpufortrt_copy(void* hostptr,size_t num_bytes,bool never_deallocate) {
  return ::run_use_increment_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_copy,
    true,
    gpufortrt_async_noval,
    never_deallocate,
    gpufortrt_counter_dynamic);
}
void* gpufortrt_copy_async(void* hostptr,size_t num_bytes,int async_arg,bool never_deallocate) {
  return ::run_use_increment_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_copy,
    false,
    async_arg,
    never_deallocate,
    gpufortrt_counter_dynamic);
}

namespace {
  template <bool update_host,bool update_section,bool blocking>
  void update(
      void* hostptr,
      int num_bytes,
      bool condition,
      bool if_present,
      int async_arg) {
    LOG_INFO(1,((update_host) ? "update_host" : "update_device")
            << ((update_section) ? "_section" : "")
            << ((!blocking) ? "_async" : "")
            << "; hostptr:"<<hostptr 
            << ", num_bytes:"<<num_bytes 
            << ", condition:"<<condition 
            << ", if_present:"<<if_present
            << ((!blocking) ? ", async_arg:" : "")
            << ((!blocking) ? std::to_string(async_arg).c_str() : ""))
    if ( condition ) {
      if ( !gpufortrt::internal::initialized ) LOG_ERROR("update_host: runtime not initialized")
      if ( hostptr == nullptr ) {
        #if NULLPTR_MEANS_NOOP
        throw std::invalid_argument("update_host: hostptr is nullptr")
        #endif
        return;
      }
      bool success = false;
      size_t loc = -1;
      if ( update_section ) {
        loc = gpufortrt::internal::record_list.find_record(hostptr,num_bytes,success/*inout*/); 
      } else {
        loc = gpufortrt::internal::record_list.find_record(hostptr,success/*inout*/); 
      }
      if ( !success && !if_present ) { 
        LOG_ERROR("update_host: no record found for hostptr="<<hostptr)
      } else if ( success ) {
        auto& record = gpufortrt::internal::record_list.records[loc];
        gpufortrt_queue_t queue = gpufortrt_default_queue; 
        if ( !blocking ) {
          queue = gpufortrt::internal::queue_record_list.use_create_queue(async_arg);
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
    int async_arg) {
  ::update<false,false,true>(hostptr,0,condition,if_present,async_arg); 
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
    int async_arg) {
  ::update<false,true,true>(hostptr,num_bytes,condition,if_present,async_arg); 
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
    int async_arg) {
  ::update<false,false,false>(hostptr,0,condition,if_present,async_arg); 
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
    int async_arg) {
  ::update<false,false,false>(hostptr,num_bytes,condition,if_present,async_arg); 
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
      auto queue = gpufortrt::internal::queue_record_list.use_create_queue(wait_arg[i]);
      HIP_CHECK(hipStreamSynchronize(queue)) // TODO backend specific, externalize
    }
  }
}
void gpufortrt_wait_async(int* wait_arg,int num_wait,
                          int* async_arg,int num_async,
                          bool condition) {
  if ( condition ) {
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
                              bool condition) {
  if ( condition ) {
    hipEvent_t event;// TODO backend specific, externalize
    HIP_CHECK(hipEventCreateWithFlags(&event,hipEventDisableTiming))// TODO backend specific, externalize
    HIP_CHECK(hipEventRecord(event,gpufortrt_default_queue))// TODO backend specific, externalize
    for (int j = 0; j < num_async; j++) {
      auto queue_async = gpufortrt::internal::queue_record_list.use_create_queue(async_arg[j]);
      HIP_CHECK(hipStreamWaitEvent(queue_async,event,0)) // TODO backend specific, externalize
    }
  }
}
  
void* gpufortrt_deviceptr(void* hostptr) {
  auto* record = gpufortrt::internal::structured_region_stack.find_in_current_region(hostptr,1/*num_bytes*/);
  if ( record == nullptr ) {
    bool success = false;
    size_t loc = gpufortrt::internal::record_list.find_record(hostptr,1/*num_bytes*/,success/*inout*/); 
        // success: hostptr in [record.hostptr,record.hostptr+record.num_bytes)
    if ( success ) {
      record = &gpufortrt::internal::record_list.records[loc];
    }
  }
  if ( record != nullptr ) {
    size_t offset_bytes;
    record->is_host_data_subset(hostptr,1/*num_bytes*/,offset_bytes/*inout*/);
    return static_cast<void*>(static_cast<char*>(record->deviceptr) + offset_bytes);
  } else {
    return nullptr;
  } 
}