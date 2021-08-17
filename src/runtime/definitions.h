#ifndef __DEFINITIONS_H__
#define __DEFINITIONS_H__

#ifndef MAX_QUEUES
#define MAX_QUEUES 64
#endif

#ifndef INITIAL_RECORDS_CAPACITY
#define INITIAL_RECORDS_CAPACITY 4096
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif


enum GpufortMapKind {
  GPUFORT_MAP_UNDEFINED = 0,   //< Not defined (initial value).
  GPUFORT_MAP_ALLOC,           //< If not already present, allocate.
  GPUFORT_MAP_TO,              //< ..., and copy to device.
  GPUFORT_MAP_FROM,            //< ..., and copy from device.
  GPUFORT_MAP_TOFROM,          //< ..., and copy to and from device.
  GPUFORT_MAP_PRESENT,         //< Must already be present.
  GPUFORT_MAP_RELEASE,         //< Decrement usage count and deallocate if zero.
  GPUFORT_MAP_DELETE,          //< Deallocate a mapping, without copying from device.
  GPUFORT_MAP_FORCE_ALLOC,     //< Allocate.
  GPUFORT_MAP_FORCE_TO,        //< ..., and copy to device.
  GPUFORT_MAP_FORCE_FROM,      //< ..., and copy from device.
  GPUFORT_MAP_FORCE_TOFROM,    //< ..., and copy to and from device.
  GPUFORT_MAP_FORCE_DEVICEPTR, //< Is a device pointer.
  GPUFORT_MAP_DEVICE_RESIDENT, //< Do not map, copy bits for firstprivate instead. OpenACC device_resident.
  GPUFORT_MAP_LINK,            //< OpenACC link.
  GPUFORT_MAP_FIRSTPRIVATE,    //< Allocate per gang.
  GPUFORT_MAP_USE_DEVICE_PTR   //< Return device pointer for given host pointer.
};

#endif // __DEFINITIONS_H__
