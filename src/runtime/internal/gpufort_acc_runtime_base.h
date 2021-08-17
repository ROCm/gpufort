#ifndef __GPUFORT_ACC_RUNTIME_BASE_H__
#define __GPUFORT_ACC_RUNTIME_BASE_H__

#include "definitions.h"
#include "record.h"
#include "queue.h"

#ifndef MAX_QUEUES
#define MAX_QUEUES 64
#endif

#ifndef INITIAL_RECORDS_CAPACITY
#define INITIAL_RECORDS_CAPACITY 4096
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif


namespace internal {
  std::vector<Record> records(INITIAL_RECORDS_CAPACITY);
  std::vector<Queue>  queues(MAX_QUEUES);
}

extern "C" {
  
}

#endif // __GPUFORT_ACC_RUNTIME_BASE_H__
