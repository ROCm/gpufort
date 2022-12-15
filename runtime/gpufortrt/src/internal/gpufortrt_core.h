// SPDX-License-Identifier: MIT
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#ifndef GPUFORTRT_CORE_H
#define GPUFORTRT_CORE_H

#include <iostream>
#include <vector>
#include <tuple>
#include <cstddef>

#include "gpufortrt_types.h"

#include "auxiliary.h"

namespace gpufortrt {
  namespace internal {
    /**
     * \note: Data layout must match that of Fortran `record_t` type!
     */
    struct record_t {
      int id = -1; //> Id of this record. Negative id means that the record is not initialized.
      void* hostptr = nullptr; //> Typically points to an address in host memory.
      void* deviceptr = nullptr; //> typically points to an address in device memory.
      std::size_t num_bytes = 0;   //> Number of mapped bytes.
      std::size_t reserved_bytes = 0; //> Device buffer size, might be larger than `num_bytes`.
      int struct_refs = 0; //> Structured references counter.
      int dyn_refs = 0;    //> Dynamic references counter.

    public:
      /** Write string representation of 
       * this record to the ostream. */
      void to_string(std::ostream& os) const;
      /** 
       * If this record is initalized. 
       * If a record is initialized it can be used or released.
       * \note If the record is initialized, this does not necessary mean that
       *       device data has been allocated for it.
       *       This is for example not the case if a zero-length array has been
       *       mapped onto the device.
       * */
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
      bool can_be_destroyed(const int struct_ref_threshold = 0) const;
      /**Increments specified counter.*/
      void inc_refs(const gpufortrt_counter_t ctr);
      /**Decrements specified counter.*/
      void dec_refs(const gpufortrt_counter_t ctr);
      
      /** Copy host data to device. */
      void copy_to_device(
        const bool blocking,
        gpufortrt_queue_t queue);
      void copy_section_to_device(
        void* hostptr,
        const std::size_t num_bytes,
        const bool blocking,
        gpufortrt_queue_t queue);
      /** Copy device data to host. */
      void copy_to_host(
        const bool blocking,
        gpufortrt_queue_t queue);
      void copy_section_to_host(
        void* hostptr,
        const std::size_t num_bytes,
        const bool blocking,
        gpufortrt_queue_t queue);

      /**
       * \return Tuple: If the host data byte address `hostptr`
       *         fits into the host data region associated with this record (entry 0),
       *         and the offset in bytes between `hostptr` and the record's `hostptr` member.
       * \note If the record's host data uses 0 bytes, the `hostptr` must point
       *       to the record's `hostptr`. Otherwise, `false` is returned.
       */
      std::tuple<bool,std::ptrdiff_t> is_host_data_element(void* hostptr) const;

      /**
       * \return Tuple: If the host data byte address `hostptr`
       *         fits into the host data region associated with this record (entry 0),
       *         and the offset in bytes between `hostptr` and the record's `hostptr` member.
       * \note If the record's host data uses 0 bytes, the `hostptr` must point
       *       to the record's hostptr and `num_bytes` must be 0. Otherwise, `false` is returned.
       * \param[in] num_bytes Number of bytes of the searched region,
       *            must be at least 1. Otherwise, an exception is thrown.
       */
      std::tuple<bool,std::ptrdiff_t> is_host_data_subset(void* hostptr,std::size_t num_bytes) const;
      
      /* Setup this record. Constructor. */
      void setup(
        const int id,
        void* hostptr,
        const std::size_t num_bytes,
        const bool allocate_device_buffer,
        const bool copy_to_device,
        const bool blocking,
        gpufortrt_queue_t queue,
        const bool reuse_existing);
      
      /* Release this record, i.e. allocated device buffers
       * can be repurposed. */
      void release();

      /* Destroy this record. */
      void destroy();

      /** Decrement the specified reference counter,
       *  release the record if all counters are zero.
       *  Perform a copy out operation if specified.
       *  \param[in] finalize sets the dynamic reference counter to zero.
       *             An exception is thrown if the any other counter is specified.
       */
      void decrement_release(
        const gpufortrt_counter_t ctr_to_update,
        const bool copyout, const bool finalize,
        const bool blocking, gpufortrt_queue_t queue);
    };

    struct record_list_t {
      std::vector<record_t> records;
      int last_record_index = 0;
      std::size_t total_memory_bytes = 0;
    public:
       /**
        * \note Not thread safe
        */
       record_t& operator[](const int i);
       
       /**
        * \note Not thread safe
        */
       const record_t& operator[](const int i) const;

       /**
        * \note Not thread safe
        */
       void reserve(const int capacity);
       
       /** 
        * Write string representation of 
        * this record to the ostream. 
        */
       void to_string(std::ostream& os) const;
       
       /**
        * \note Not thread safe
        */
       void destroy();
    
       /**
        * Finds a record that carries the given hostptr and returns the location.
        *
        * \note Not thread safe
        * \return Tuple: If the host address `hostptr` could be associated with a record (entry 0),
        *         the location in the list (entry 1)
        *         and the offset in bytes between `hostptr` and the record's `hostptr` member (entry 2).
        * 
        * \note Not thread safe
        */
       std::tuple<bool,std::size_t,std::ptrdiff_t> find_record(void* hostptr) const;
       
       /**
        * Finds a record whose associated host data fits the data
        * section described by `hostptr` and `num_bytes`.
        *
        * \param[in] num_bytes Number of bytes of the searched region.
        * \return Tuple: If the host address `hostptr` could be associated with a record (entry 0),
        *         the location in the list (entry 1)
        *         and the offset in bytes between `hostptr` and the record's `hostptr` member (entry 2).
        * 
        * \note Not thread safe
        */
       std::tuple<bool,std::size_t,std::ptrdiff_t> find_record(void* hostptr,std::size_t num_bytes) const;

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
        *
        * \return Tuple: Location of the record (entry 0) and if 
        *         an existing record and its device buffer can be reused (entry 1).
        */
       std::tuple<std::size_t,bool> find_available_record(const std::size_t num_bytes);

       /**
        * Increments a record's reference counter.
        * \param[in] check_restrictions check the data clause restrictions
        * \return Tuple: If a record associated with `hostptr` could be found (entry 0),
        *         and location of the record in the list (entry 1).
        * \throw std::invalid_argument if the data clause restrictions are violated, if not specified otherwise.
        * \note Not thread safe.
        */
       std::tuple<bool,std::size_t> increment_record_if_present(
         const gpufortrt_counter_t ctr_to_update,
         void* hostptr,
         const std::size_t num_bytes,
         const bool check_restrictions);
     
       /**
        * Creates a record (inclusive the host-to-device memcpy where required) or increments a record's
        * reference counter.
        *
        * \param[in] never_deallocate Initializes records with structured reference counter value "1".
        * \note Not thread safe.
        * \throw std::invalid_argument if the data clause restrictions are violated.
        */
       std::size_t create_increment_record(
         const gpufortrt_counter_t ctr_to_update,
         void* hostptr,
         const std::size_t num_bytes,
         const bool never_deallocate,
         const bool allocate_device_buffer,
         const bool copy_to_device,
         const bool blocking,
         gpufortrt_queue_t queue);

      /**
       * Decrements a record's reference counter and destroys the record if
       * structured and dynamic reference counter are zero. Copies the data to the host
       * before destruction if specified.
       * 
       * \note Not thread safe.
       * \throw std::invalid_argument if the data clause restrictions are violated.
       */
       void decrement_release_record(
         const gpufortrt_counter_t ctr_to_update,
         void* hostptr,
         const std::size_t num_bytes,
         const bool copyout,
         const bool finalize,
         const bool blocking,
         gpufortrt_queue_t queue);
    };

    struct structured_region_stack_entry_t {
      int region_id = -1; //> Identifier of a structured region.
      gpufortrt_map_kind_t map_kind = gpufortrt_map_kind_undefined; //> Kind of mapping.
      void* hostptr = nullptr; //> Typically points to an address in host memory.
      std::size_t num_bytes = 0;   //> Number of mapped bytes.
      record_t* record = nullptr; //> Points to a record in the record list, or is nullptr.
    public:
      structured_region_stack_entry_t(const int region_id,
                                      const gpufortrt_map_kind_t map_kind,
                                      void* hostptr,
                                      const std::size_t num_bytes);
      structured_region_stack_entry_t(const int region_id,
                                      const gpufortrt_map_kind_t map_kind,
                                      record_t* record);
      /** Write string representation to the ostream. */
      void to_string(std::ostream& os) const;
    };

    struct structured_region_stack_t {
      int current_region = 0;
      std::vector<structured_region_stack_entry_t> entries;
    public:
      void reserve(const int capacity);
      
      /** 
       * Decrement the structured reference counter of 
       * all records associated with the current structured
       * region and then remove them from the stack.
       */ 
      void enter_structured_region();
      
      /**
       * Push a new stack entry without associated record to stack
       * and associate it with the current region.
       */
      void push_back(const gpufortrt_map_kind_t map_kind,
                     void* hostptr,
                     const std::size_t num_bytes);
      
      /**
       * Push a new record on the stack 
       * and associate it with the current region.
       */
      void push_back(const gpufortrt_map_kind_t map_kind,
                     record_t* record);
      
      /**
       * Find an entry in the stack that is associated with `hostptr`.
       * \return Tuple: Flag indicating success (entry 0). A pointer to a record that contains the data that the `hostptr`
       *         points to, or `nullptr` (entry 1).
       *         Offset with respect to the found record's host data (entry 2).
       *         Flag indicating that a `no_create` mapping was applied to `hostptr` but 
       *         no record exists for `hostptr` (entry 3).
       */
      std::tuple<
        bool,
        gpufortrt::internal::record_t*,
        std::ptrdiff_t,
        bool> 
            find_record(void* hostptr) const;

      /** 
       * Decrement the structured reference counter of 
       * all records associated with the current structured
       * region and then remove them from the stack.
       */ 
      void leave_structured_region(const bool blocking,
                                   gpufortrt_queue_t queue);
    };

    struct queue_record_t {
      int id = -1;
      gpufortrt_queue_t queue = gpufortrt_default_queue;
    public:
      void to_string(std::ostream& os) const;
      /** Initialize this queue record,
       * create a new queue. */
      void setup(const int id);
      /** Destroy this queue record,
       * set the `id` to negative value. */
      void destroy();
      /** \return If this queue is initialized or destroyed. */
      bool is_initialized() const;

      /** \return If all operations in this queue
       * have been completed.*/
      bool test();

      /** Synchronize the queue. */
      void synchronize();
    };

    struct queue_record_list_t {
      std::vector<queue_record_t> records;
    private:
      /** Find queue record with the given `id`.
       * \param[inout] success The returned index is only valid if success indicates 'true'.
       */
      std::tuple<bool,std::size_t> find_record(const int id) const;
    public:
      queue_record_t& operator[](const int i);
      const queue_record_t& operator[](const int i) const;
     
      size_t size();
      /** Reserve space for `capacity` queues. */
      void reserve(const int capacity);

      /** Destroy all existing queues. */
      void destroy();
      /**
       * Use an existing queue with identifier `id` or
       * create one for that identifier and return it.
       *
       * \return a queue if `id` is greater than 0, or `nullptr`.
       */
      gpufortrt_queue_t use_create_queue(const int id);
      
      /** \return If all operations in the queue with identifier `id` 
       * have been completed.*/
      bool test(const int id);

      /** Synchronize queue with identifier `id`. */
      void synchronize(const int id);
      
      /** \return If all operations in all queues
       * have been completed.*/
      bool test_all();

      /** Synchronize all queues. */
      void synchronize_all();
    };

    // global parameters, influenced by environment variables
    extern std::size_t INITIAL_QUEUE_RECORDS_CAPACITY;//= 128
    extern std::size_t INITIAL_RECORDS_CAPACITY;//= 4096
    extern std::size_t INITIAL_STRUCTURED_REGION_STACK_CAPACITY;//= 128
    // reuse/fragmentation controls
    extern int BLOCK_SIZE;            // = 32
    extern double REUSE_THRESHOLD;    // = 0.9 //> only reuse record if mem_new>=factor*mem_old
    extern int NUM_REFS_TO_DEALLOCATE;// = -5  //> dealloc device mem only if struct_refs takes this value

    // global variables
    extern bool initialized;
    extern int default_async_arg;
    extern std::size_t num_records;
    extern record_list_t record_list;
    extern queue_record_list_t queue_record_list; 
    extern structured_region_stack_t structured_region_stack;

    /** If the map_kind implies that a device buffer must be allocated. */
    bool implies_allocate_device_buffer(
            const gpufortrt_map_kind_t map_kind,
            const gpufortrt_counter_t ctr);
    /** If the map_kind implies that the host data must be copied to the device. */
    bool implies_copy_to_device(const gpufortrt_map_kind_t map_kind);
    /** If the map_kind implies that the device data must be copied to the host
     *  at destruction. */
    bool implies_copy_to_host(const gpufortrt_map_kind_t map_kind);
   
    template<typename T>
    std::tuple<bool,std::ptrdiff_t> is_host_data_element(const T& t,void* hostptr) {
      bool result = false;
      std::ptrdiff_t offset = static_cast<char*>(hostptr) - static_cast<char*>(t.hostptr);
      if ( t.num_bytes > 0 ) {
        result = offset >= 0 && offset < t.num_bytes;    
      } else { // If this record maps a zero-size memory block
        result = offset == 0;
      }
      return std::make_tuple(result, offset);
    }
    
    template<typename T>
    std::tuple<bool,std::ptrdiff_t> is_host_data_subset(const T& t,void* hostptr, std::size_t num_bytes) {
      bool result = false;
      std::ptrdiff_t offset = static_cast<char*>(hostptr) - static_cast<char*>(t.hostptr);
      if ( t.num_bytes > 0 ) {
        result = offset >= 0 && (offset + num_bytes <= t.num_bytes);    
      } else { // If this record maps a zero-size memory block, the input must be a zero-size memory block with the same address.
        result = offset == 0 && num_bytes == 0;
      }
      return std::make_tuple(result, offset);
    }
    
    /** Offset a record's `deviceptr` by `offset`.*/
    template<typename T>
    void* offsetted_record_deviceptr(const T& t,std::ptrdiff_t offset) {
      return static_cast<void*>(static_cast<char*>(t.deviceptr) + offset);
    }

    /** Offset a record's `deviceptr` by the different of argument `hostptr`
     * to the record's `hostptr`.*/
    template<typename T>
    void* offsetted_record_deviceptr(const T& t,void* hostptr) {
      auto element_tuple/*is_element,offset*/ = gpufortrt::internal::is_host_data_element(t,hostptr);
      return offsetted_record_deviceptr(t,std::get<1>(element_tuple));
    }


  } // namespace internal
} // namespace gpufortrt

std::ostream& operator<<(std::ostream& os,const gpufortrt::internal::record_t& record);
std::ostream& operator<<(std::ostream& os,const gpufortrt::internal::structured_region_stack_entry_t& entry);
std::ostream& operator<<(std::ostream& os,const gpufortrt::internal::queue_record_t& queue_record);
#endif // GPUFORTRT_CORE_H
