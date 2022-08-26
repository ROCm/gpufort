// SPDX-License-Identifier: MIT
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#ifndef OPENACC_LIB_H
#define OPENACC_LIB_H
#ifdef __cplusplus
extern "C" {
#endif
#include "openacc.h"

#define h_void void
#define d_void void

// except for acc_on_device all routines are only available on the host

int acc_get_num_devices(acc_device_t dev_type);

void acc_set_device_type(acc_device_t dev_type);
acc_device_t acc_get_device_type(void);

void acc_set_device_num(int dev_num, acc_device_t dev_type);
int acc_get_device_num(acc_device_t dev_type);

size_t acc_get_property(int dev_num,
                        acc_device_t dev_type,
                        acc_device_property_t property);
const 
char* acc_get_property_string(int dev_num,
                              acc_device_t dev_type,
                              acc_device_property_t property);

void acc_init(acc_on_device_t dev_type);
void acc_shutdown(acc_device_t dev_type);

int acc_async_test(int wait_arg);
int acc_async_test_device(int wait_arg, int dev_num);
int acc_async_test_all(void);
int acc_async_test_all_device(int dev_num);

void acc_wait(int wait_arg);
void acc_wait_device(int wait_arg, int dev_num);
void acc_wait_async(int wait_arg, int async_arg);
void acc_wait_device_async(int wait_arg, int async_arg,
int dev_num);
void acc_wait_all(void);
void acc_wait_all_device(int dev_num);
void acc_wait_all_async(int async_arg);
void acc_wait_all_device_async(int async_arg, int dev_num);

int acc_get_default_async(void);
void acc_set_default_async(int async_arg);

#ifdef __HIPCC__
__host__ __device__ int acc_on_device(acc_device_t dev_type);
#else
int acc_on_device(acc_device_t dev_type);
#endif

d_void* acc_malloc(size_t bytes); // TODO create gpufortrt equivalent
void acc_free(d_void* data_dev);  // TODO create gpufortr equivalent

d_void* acc_copyin(h_void* data_arg, size_t bytes);
void acc_copyin_async(h_void* data_arg, size_t bytes,
                      int async_arg);

d_void* acc_create(h_void* data_arg, size_t bytes);
void acc_create_async(h_void* data_arg, size_t bytes,
                      int async_arg);

void acc_copyout(h_void* data_arg, size_t bytes);
void acc_copyout_async(h_void* data_arg, size_t bytes,
                       int async_arg);
void acc_copyout_finalize(h_void* data_arg, size_t bytes);
void acc_copyout_finalize_async(h_void* data_arg, size_t bytes,
                                int async_arg);

void acc_delete(h_void* data_arg, size_t bytes);
void acc_delete_async(h_void* data_arg, size_t bytes,
                      int async_arg);
void acc_delete_finalize(h_void* data_arg, size_t bytes);
void acc_delete_finalize_async(h_void* data_arg,
                               size_t bytes, int async_arg);

void acc_update_device(h_void* data_arg, size_t bytes);
void acc_update_device_async(h_void* data_arg, size_t bytes,
                             int async_arg);

void acc_update_self(h_void* data_arg, size_t bytes);
void acc_update_self_async(h_void* data_arg, size_t bytes,
                           int async_arg);

void acc_map_data(h_void* data_arg, d_void* data_dev,
                  size_t bytes);
void acc_unmap_data(h_void* data_arg);

d_void* acc_deviceptr(h_void* data_arg);

h_void* acc_hostptr(d_void* data_dev);

int acc_is_present(h_void* data_arg, size_t bytes);

void acc_memcpy_to_device(d_void* data_dev_dest,
                          h_void* data_host_src, size_t bytes);
void acc_memcpy_to_device_async(d_void* data_dev_dest,
                                h_void* data_host_src, size_t bytes,
                                int async_arg);

void acc_memcpy_from_device(h_void* data_host_dest,
                            d_void* data_dev_src, size_t bytes);
void acc_memcpy_from_device_async(h_void* data_host_dest,
                                  d_void* data_dev_src, size_t bytes,
                                  int async_arg)

void acc_attach(h_void** ptr_addr);
void acc_attach_async(h_void** ptr_addr, int async_arg);

void acc_detach(h_void** ptr_addr);
void acc_detach_async(h_void** ptr_addr, int async_arg);
void acc_detach_finalize(h_void** ptr_addr);
void acc_detach_finalize_async(h_void** ptr_addr,
                               int async_arg);

void acc_memcpy_d2d(h_void* data_arg_dest,
                    h_void* data_arg_src, size_t bytes,
                    int dev_num_dest, int dev_num_src);

void acc_memcpy_d2d_async(h_void* data_arg_dest,
                          h_void* data_arg_src, size_t bytes,
                          int dev_num_dest, int dev_num_src,
                          int async_arg_src);

#ifdef __cplusplus
} // extern "C"
#endif
#endif // OPENACC_LIB_H
