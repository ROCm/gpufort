#ifndef GPUFORT_LOOP_H
#define GPUFORT_LOOP_H
#include <limits>

namespace gpufort {
  constexpr int acc_resource_all = std::numeric_limits<int>::max();

  struct acc_coords {
    int gang;
    int worker;
    int vector_lane;
    __host__ __device__ __forceinline__ acc_coords(
          int gang=1,
          int worker=1,
          int vector_lane=1) :
        gang(gang),
        worker(worker),
        vector_lane(vector_lane) {}    

    __host__ __device__ bool operator< (const gpufort::acc_grid& grid) const {
        return    this->vector_lane < grid.vector_lanes 
               && this-worker < grid.workers
               && this->gang < grid.gangs;
    }
   
    /** \return gang member of this struct. */ 
    __host__ __device__ __forceinline gang_id() {
        return this->gang;
    }
    
    /** \return global ID for the worker with respect to the number of workers per gang. */
    __host__ __device__ __forceinline worker_id(int num_workers) {
        return this->worker + this->gang*num_workers;
    }
    
    /** \return global ID for the vector lane with respect to the number of workers,
        per gang and the vector-length per worker. */
    __host__ __device__ __forceinline vector_lane_id(int vector_length,
                                                     int num_workers) {
        return this->vector_lane +
                vector_length*
                (this->worker + this->gang*num_workers);
    }

    /** Overloaded variant that takes the worker member 
        from the grid triple. */
    __host__ __device__ __forceinline worker_id(
        const gpufort::acc_res& grid) {
        return this->vector_lane_id(grid.workers);
    }
  
    /** Overloaded variant that takes the worker and vector_lane member 
        from the grid triple. */
    __host__ __device__ __forceinline vector_lane_id(
        const gpufort::acc_res& grid) {
        return this->vector_lane_id(grid.vector_lanes,grid.workers);
    }
  };
  
  struct acc_grid {
    int gangs;   //< Total number of gangs.
    int workers; //< Workers per gang.
    int vector_lanes; //< Vector lanes per worker.
    __host__ __device__ __forceinline__ acc_grid(
          int gangs=1,
          int workers=1,
          int vector_lanes=1) :
        gangs(gangs),
        workers(workers),
        vector_lanes(vector_lanes) {}    
   
    /** :return: the product of all components.*/
    __host__ __device__ __forceinline__ int total_num_vector_lanes() const {
      return this->gangs * this->workers * this->vector_lanes;
    }
    
    __host__ __device__ __forceinline__ int total_num_workers() const {
      return this->gangs * this->workers;
    }
    
    __host__ __device__ __forceinline__ int total_num_gangs() const {
      return this->gangs;
    }
  }; // acc_grid

  __host__ __device__ __forceinline__ int div_round_up(int x, int y) {
    return ((x) / (y) + ((x) % (y) != 0));
  }
  
  /**
   * Checks if `idx` is less or equal to the last index of the loop iteration
   * range.
   *
   * \note Takes only the sign of `step` into account, not its value.
   *
   * \param[in] idx loop index
   * \param[in] last last index of the loop iteration range
   * \param[in] step step size of the loop iteration range
   */
  __host__ __device__ __forceinline__ bool loop_cond(int idx,int last,int step=1) {
    return (step>0) ? ( idx <= last ) : ( idx >= last ); 
  }
  
  /**
   * Number of iterations of a loop that runs from 'first' to 'last' (both inclusive)
   * with step size 'step'. Note that 'step' might be negative and 'first' > 'last'.
   * If 'step' lets the loop run into a different direction than 'first' to 'last', this function 
   * returns 0.
   *
   * \param[in] first first index of the loop iteration range
   * \param[in] last last index of the loop iteration range
   * \param[in] step step size of the loop iteration range
   */
  __host__ __device__ __forceinline__ int loop_len(int first,int last,int step=1) {
    const int len_minus_1 = (last-first) / step;
    return ( len_minus_1 >= 0 ) ? (1+len_minus_1) : 0; 
  }
  
  /**
   * Variant of outermost_index that takes the length of the loop
   * as additional argument.
   *
   * \param[in] first first index of the outermost loop iteration range
   * \param[in] len the loop length.
   * \param[in] step step size of the outermost loop iteration range
   */
  __host__ __device__ __forceinline__ int outermost_index(
    int& collapsed_idx,
    int& collapsed_len,
    const int first, const int len, const int step = 1
  ) {
    collapsed_len /= len;
    const int idx = collapsed_idx / collapsed_len; // rounds down
    collapsed_idx -= idx*collapsed_len;
    return (first + step*idx);
  }
} // gpufort

#endif # GPUFORT_LOOP_H
