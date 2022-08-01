#include "gpufortrt_core.h"

// global parameters, influenced by environment variables
int gpufortrt::LOG_LEVEL = 0;
int gpufortrt::MAX_QUEUES = 64;
int gpufortrt::INITIAL_RECORDS_CAPACITY = 4096;
int gpufortrt::BLOCK_SIZE = 32;
double gpufortrt::REUSE_THRESHOLD = 0.9;
int gpufortrt::NUM_REFS_TO_DEALLOCATE = -5;

// global variables
size_t gpufortrt::record_counter = 0;
record_list_t gpufortrt::record_list;
std::vector<queue_t> gpufortrt::queues;