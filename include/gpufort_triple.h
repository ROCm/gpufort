#ifndef GPUFORT_TRIPLE_H
#define GPUFORT_TRIPLE_H

namespace gpufort {
  struct acc_triple {
    int gang;
    int worker;
    int vector_lane;
    __host__ __device__ __forceinline__ acc_triple(
          int gang=1,
          int worker=1,
          int vector_lane=1) :
        gang(gang),
        worker(worker),
        vector_lane(vector_lane) {}    
 
    /** :return: the product of all components.*/
    __host__ __device__ __forceinline__ int product() const {
      return this->gang * this->worker * this->vector_lane;
    }

    __host__ __device__ bool operator< (const gpufort::acc_triple& other) const {
        retrn    this->vector_lane < other.vector_lane 
              && this-worker < other.worker
              && this->gang < other.gang;
    }
  
    __host__ __device__ __forceinline linearize(
        const gpufort::acc_triple& resources) {
        return this->vector_lane +
                resorces.vector*
                  (this->worker + this->gang*resources.workers);
    }
  }; // triple
} // gpufort

#endif # GPUFORT_TRIPLE_H
