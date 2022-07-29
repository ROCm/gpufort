#include <iostream>

namespace gpufortrt {

/**
 * \note: Data layout must match that of Fortran `map_kind` type!
 * \note: Upper case first letter used because `delete` is C++ keyword.
 */
enum class map_kind_t {
  Dec_struct_refs = -1,
  Undefined       = 0,
  Present         = 1,
  Delete          = 2,
  Create          = 3,
  No_create       = 4,
  Copyin          = 5,
  Copyout         = 6,
  Copy            = 7
};

/**
 * Reference counter type.
 */
enum class counter_t {
  Undefined = -1,
  Structured = 0, //< Structured references (structured data regions, compute constructs)
  Dynamic = 1     //< Dynamic references (enter/exit data)
}
  
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
  mapkind_t map_kind    = map_kind_t::Undefined

public:
  //void print();
  /** Write string representation of 
   * this record to the ostream. */
  void to_string(std::ostream& os);
  /** If data is allocated for this record. If a 
   * record is initialized it can be used or released. */
  bool isinitialized();
  /** If the records' device data is used, i.e. if 
   * any of the counter is positive. */
  bool isused();
  /** If the records' device data is allocated
   * but the memory can be repurposed. */ 
  bool isreleased();
  /** Copy host data to device. */
  void copy_to_device();
  /** Copy device data to host. */
  void copy_to_host();
  bool issubarray();
  /* Release this record, i.e. allocated device buffers
   * can be repurposed. */
  void release();
  /* Setup this record. Constructor. */
  void setup();
  /* Destroy this record. */
  void destroy();
  /**Increments specified counter.*/
  void inc_refs(counter_t ctr);
  /**Decrements specified counter.*/
  void dec_refs(counter_t ctr);
  /**
   * \return If the structured references counter is less or
   * equal than a certain non-positive threshold and all other counters are 0.
   */
  bool can_be_destroyed(int struct_ref_threshold = 0);
};

/**
 * \note: Data layout must match that of Fortran `queue_t` type!
 */
struct queue_t {
  void* queueptr = nullptr;
};

} // namespace gpufortrt
