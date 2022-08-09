// SPDX-License-Identifier: MIT
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include <iostream>
#include <vector>

#include "gpufortrt_api.h"

#define HIP_CHECK(condition)         \
  {                                  \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
  }

std::ostream& operator<<(std::ostream& os, gpufortrt_map_kind_t map_kind);

namespace gpufortrt {
  namespace internal {
    /**
     * \note: Data layout must match that of Fortran `record_t` type!
     */
    struct record_t {
      int id                = -1;
      void* hostptr         = nullptr;
      void* deviceptr       = nullptr;
      size_t num_bytes      = 0;
      size_t num_bytes_used = 0;
      int struct_refs       = 0;
      int dyn_refs          = 0;
      gpufortrt_map_kind_t map_kind = gpufortrt_map_kind_undefined;

    public:
      /** Write string representation of 
       * this record to the ostream. */
      void to_string(std::ostream& os) const;
      /** If data is allocated for this record. If a 
       * record is initialized it can be used or released. */
      bool is_initialized() const;
      /** If the records' device data is used, i.e. if 
       * any of the counter is positive. */
      bool is_used() const;
      /** If the records' device data is allocated
       * but the memory can be repurposed. */ 
      bool is_released() const;
      /**
       * \return If the released device buffer was not reused a number
       * of times (`threshold`).
       *
       * If the structured references counter is less or
       * equal than a certain non-positive threshold and all other counters are 0.
       */
      bool can_be_destroyed(int struct_ref_threshold = 0);
      /**Increments specified counter.*/
      void inc_refs(counter_t ctr);
      /**Decrements specified counter.*/
      void dec_refs(counter_t ctr);
      
      /** Copy host data to device. */
      void copy_to_device(
        bool blocking,
        gpufortrt_queue_t queue);
      /** Copy device data to host. */
      void copy_to_host(
        bool blocking,
        gpufortrt_queue_t queue);
      bool is_subarray(
        void* hostptr, size_t num_bytes, size_t& offset_bytes) const;
      
      /* Setup this record. Constructor. */
      void setup(
        int id,
        void* hostptr,
        size_t num_bytes,
        gpufortrt_map_kind_t map_kind,
        bool blocking,
        gpufortrt_queue_t queue,
        bool reuse_existing);
      
      /* Release this record, i.e. allocated device buffers
       * can be repurposed. */
      void release();

      /* Destroy this record. */
      void destroy();

      /** Decrement the specified counter,
       *  release the record if all counters are zero.
       *  Perform a copy out operation if record's map kind requires this.
       */
      void decrement_release_record(record_t& record,
                                    gpufortrt_counter_t ctr_to_update,
                                    bool blocking, gpufortrt_queue_t queue,
                                    bool finalize);
    };
    
    struct record_list_t {
      std::vector<record_t> records;
      int last_record_index = 0;
      size_t total_memory_bytes = 0;
    public:
       void reserve(int capacity);
       
       /** 
        * Write string representation of 
        * this record to the ostream. 
        */
       void to_string(std::ostream& os) const;
       
       void destroy();
    
       /**
        * Finds a record for a given host ptr and returns the location.
        *
        * \note Not thread safe
        */
       size_t find_record(void* hostptr,bool& success); // TODO why success and return value (can be negative)?

       /**
        * Searches first available record from the begin of the record search space.
        * If it does not find an available record in the current search space,
        * it takes the record right after the end of the search space
        * and grows the search space by 1.
        *
        * Tries to reuse existing records that have been released.
        * Checks how many times a record has been considered (unsuccessfully) for reusing and deallocates
        * associated device data if that has happend NUM_REFS_TO_DEALLOCATE times (or more).
        *
        * \note Not thread safe.
        */
       size_t find_available_record(size_t num_bytes,bool reuse_existing);
     
       /**
        * Inserts a record (inclusive the host-to-device memcpy where required) or increments a record's
        * reference counter.
        *
        * \note Non-alloctable and non-pointer module variables are initialized
        * with structured reference counter value "1".
        * \note Not thread safe.
        */
       size_t use_increment_record(
         void* hostptr,
         size_t num_bytes,
         gpufortrt_map_kind_t map_kind,
         counter_t ctr_to_update
         bool blocking,
         gpufortrt_queue_t queue,
         bool never_deallocate);

      /**
       * Decrements a record's reference counter and destroys the record if
       * the reference counter is zero. Copies the data to the host beforehand
       * if specified.
       * 
       * \note Not thread safe.
       */
       void decrement_release_record(
         void* hostptr,
         gpufortrt_counter_t ctr_to_update,
         bool blocking,
         gpufortrt_queue_t queue,
         bool finalize);
    };

    struct structured_region_stack_entry_t {
      int region_id = -1;
      record_t* record;
    public:
      structured_region_stack_entry_t(int region_id,record_t& record);
    }

    struct structured_region_stack_t {
      int current_region = 0;
      std::vector<structured_region_stack_entry_t> entries;
    public:
      void reserve(int capacity);
      
      /** 
       * Decrement the structured reference counter of 
       * all records associated with the current structured
       * region and then remove them from the stack.
       */ 
      void enter_structured_region();
      /**
       * Push a new record on the stack 
       * and associate it with the current region.
       */
      void push_back(record_t& record);
      

      /**
       * Find an entry in the stack that is associated with `hostptr`.
       * \return a pointer to a record that contains the data that the `hostptr`
                 points too, or nullptr.
       */
      record_t* find(void* hostptr);
      
      /**
       * Find an entry in the current structured region that is associated with `hostptr`.
       * \note: This method is specifically for retrieving
       * records associated with the implicit region
       * that is introduced around a compute region.
       * \return a pointer to a record that contains the data that the `hostptr`
                 points too, or nullptr.
       */
      record_t* find_in_current_region(void* hostptr);

      /** 
       * Decrement the structured reference counter of 
       * all records associated with the current structured
       * region and then remove them from the stack.
       */ 
      void leave_structured_region();
    }

    struct queue_record_t {
      int id;
      gpufortrt_queue_t queue;
    public:
      /** Initialize this queue record,
       * create a new queue. */
      void setup(const int id);
      /** Destroy this queue record,
       * set the `id` to negative value. */
      void destroy();

      /** If this queue is initialized or destroyed. */
      bool is_initialized() const;
    }

    struct queue_record_list_t {
      std::vector<queue_t> records;
    public:
      void reserve(int capacity);

      size_t find_record(const int id) const;
      size_t find_available_record(const int id) const;
      /**
       * Use an existing queue with identifier `id` or
       * create one for that identifier and return it.
       *
       * \return a queue if `id` is greater than 0, or `nullptr`.
       */
      gpufortrt_queue_t use_create_queue(const int id);
    }

    // global parameters, influenced by environment variables
    extern size_t INITIAL_QUEUE_RECORDS_CAPACITY;//= 128
    extern size_t INITIAL_RECORDS_CAPACITY;//= 4096
    extern size_t INITIAL_STRUCTURED_REGION_STACK_CAPACITY;//= 128
    // reuse/fragmentation controls
    extern int BLOCK_SIZE;            // = 32
    extern double REUSE_THRESHOLD;    // = 0.9 //> only reuse record if mem_new>=factor*mem_old
    extern int NUM_REFS_TO_DEALLOCATE;// = -5  //> dealloc device mem only if struct_refs takes this value

    // global variables
    extern int num_records;
    extern record_list_t record_list;
    extern queue_record_list_t queue_record_list; 
    extern structured_region_stack_t structured_region_stack; 
  } // namespace internal
}
