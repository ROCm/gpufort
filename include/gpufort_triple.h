#ifndef GPUFORT_TRIPLE_H
#define GPUFORT_TRIPLE_H

namespace gpufort {
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
        retrn    this->vector_lane < grid.vector_lanes 
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
} // gpufort

#endif # GPUFORT_TRIPLE_H
