#include <iostream>
#include <vector>

#define HIP_CHECK(condition)         \
  {                                  \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
  }

#define LOG_ERROR(level,msg) \
    std::cerr << "[gpufort][" << level << "] " << msg << std::endl;

namespace gpufortrt {
  /**
   * \note: Enum values must match those of Fortran enumeration!
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
   * \note: Data layout must match that of Fortran `counter_t` type!
   */
  enum class counter_t {
    None = 0,        
    Structured = 1, //< Structured references (structured data regions, compute constructs)
    Dynamic = 2     //< Dynamic references (enter/exit data)
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
    /** Default constructor */
    record_t() {}

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
    void copy_to_device();
    /** Copy device data to host. */
    void copy_to_host();
    bool is_subarray();
    /* Release this record, i.e. allocated device buffers
     * can be repurposed. */
    void release();
    /* Setup this record. Constructor. */
    void setup();
    /* Destroy this record. */
    void destroy();
  };

  struct record_list_t {
    std::vector<record_t> records;
    int last_record_index = 0;
    size_t total_memory_bytes = 0;
  public:
     /** 
      * Write string representation of 
      * this record to the ostream. */
     void to_string(std::ostream& os) const;

     bool is_initialized() const;
     
     void initialize();
     
     void grow();
     
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
       gpufortrt::map_kind_t map_kind,
       counter_t ctr_to_update
       bool blocking_copy,
       gpufortrt::queue_t queue,
       bool declared_module_var);

    /**
     * Decrements a record's reference counter and destroys the record if
     * the reference counter is zero. Copies the data to the host beforehand
     * if specified.
     * 
     * \note Not thread safe.
     */
     void decrement_release_record(
       void* hostptr,
       gpufortrt::counter_t ctr_to_update,
       bool veto_copy_to_host,
       bool blocking_copy,
       gpufortrt::queue_t queue);
  };

  typedef hipStream_t queue_t;
  struct queue_list_t {
    std::vector<queue_t> queues;
  }

  // global parameters, influenced by environment variables
  extern int LOG_LEVEL;               //= 0
  extern int MAX_QUEUES;              //= 64
  extern int INITIAL_RECORDS_CAPACITY;//= 4096
  ! reuse/fragmentation controls
  extern int BLOCK_SIZE;            // = 32
  extern double REUSE_THRESHOLD;       // = 0.9 //> only reuse record if mem_new>=factor*mem_old
  extern int NUM_REFS_TO_DEALLOCATE;// = -5  //> dealloc device mem only if struct_refs takes this value

  // global variables
  extern size_t record_counter;
  extern record_list_t record_list;
  extern queue_list_t queue_list; 

  // C++ API
  /**
   * Performs a copy action.
   * \return The device pointer for the given hostptr.
   */
  void* use_device_b(void* hostptr,size_t num_bytes,
                     bool condition,bool if_present);
  void* present_b(void* hostptr,size_t num_bytes
                  gpufortrt::counter_t ctr_to_update);
  void dec_struct_refs_b(void* hostptr,int async);
  void* no_create_b(void* hostptr);
  void* create_b(void* hostptr,size_t num_bytes,int async,bool declared_module_var,
                 gpufortrt::counter_t ctr_to_update);
  void delete_b(void* hostptr,finalize);
  void* copyin_b(void* hostptr,size_t num_bytes,int async,bool declared_module_var,
                 gpufortrt::counter_t ctr_to_update);
  void* copyout_b(void* hostptr,size_t num_bytes,int async,
                  gpufortrt::counter_t ctr_to_update);
  void* copy_b(void* hostptr,size_t num_bytes,int async,
               gpufortrt::counter_t ctr_to_update);
  void update_host_b(void* hostptr,bool condition,bool if_present,int async);
  void update_device_b(void* hostptr,bool condition,bool if_present,int async);
} // namespace gpufortrt
