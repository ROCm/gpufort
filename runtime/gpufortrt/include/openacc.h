// SPDX-License-Identifier: MIT
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#ifndef OPENACC_H
#define OPENACC_H
#include "gpufortrt_types.h"

extern int acc_async_noval;
extern int acc_async_sync;
extern int acc_async_default;

#define h_void void
#define d_void void

enum acc_device_t {
  acc_device_none = 0,
  acc_device_default,
  acc_device_host,
  acc_device_not_host,
  acc_device_current,
  acc_device_hip = acc_device_not_host,
  acc_device_radeon = acc_device_hip,
  acc_device_nvidia = acc_device_hip
};

enum acc_property_t {
  acc_property_memory = 0,//>integer,  size of device memory in bytes
  acc_property_free_memory,//>integer,  free device memory in bytes
  acc_property_shared_memory_support,//>integer,  nonzero if the specified device supports sharing memory with the local thread
  acc_property_name,//>string, device name*/
  acc_property_vendor,//>string, device vendor*/
  acc_property_driver,//>string, device driver version*/
};

const char* ENVIRON_VAR_ACC_DEVICE_TYPE = "ACC_DEVICE_TYPE"; //> Upper case identifer, maps to `acc_device_t` suffix.
const char* ENVIRON_VAR_ACC_DEVICE_NUM = "ACC_DEVICE_NUM"; //> Non-negative integer, number of default devices.
//const char* ENVIRON_VAR_ACC_PROFLIB = "ACC_PROFLIB";

#endif // OPENACC_H
