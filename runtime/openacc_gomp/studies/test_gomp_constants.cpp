// SPDX-License-Identifier: MIT                                                
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include <iostream>

/* One byte.  */
#define GOMP_MAP_LAST			(1 << 8)

#define GOMP_MAP_FLAG_TO		(1 << 0)
#define GOMP_MAP_FLAG_FROM		(1 << 1)
/* Special map kinds, enumerated starting here.  */
#define GOMP_MAP_FLAG_SPECIAL_0		(1 << 2)
#define GOMP_MAP_FLAG_SPECIAL_1		(1 << 3)
#define GOMP_MAP_FLAG_SPECIAL_2		(1 << 4)
#define GOMP_MAP_FLAG_SPECIAL		(GOMP_MAP_FLAG_SPECIAL_1 \
					 | GOMP_MAP_FLAG_SPECIAL_0)
/* Flag to force a specific behavior (or else, trigger a run-time error).  */
#define GOMP_MAP_FLAG_FORCE		(1 << 7)

enum gomp_map_kind
  {
    /* If not already present, allocate.  */
    GOMP_MAP_ALLOC =			0,
    /* ..., and copy to device.  */
    GOMP_MAP_TO =			(GOMP_MAP_ALLOC | GOMP_MAP_FLAG_TO),
    /* ..., and copy from device.  */
    GOMP_MAP_FROM =			(GOMP_MAP_ALLOC | GOMP_MAP_FLAG_FROM),
    /* ..., and copy to and from device.  */
    GOMP_MAP_TOFROM =			(GOMP_MAP_TO | GOMP_MAP_FROM),
    /* The following kind is an internal only map kind, used for pointer based
       array sections.  OMP_CLAUSE_SIZE for these is not the pointer size,
       which is implicitly POINTER_SIZE_UNITS, but the bias.  */
    GOMP_MAP_POINTER =			(GOMP_MAP_FLAG_SPECIAL_0 | 0),
    /* Also internal, behaves like GOMP_MAP_TO, but additionally any
       GOMP_MAP_POINTER records consecutive after it which have addresses
       falling into that range will not be ignored if GOMP_MAP_TO_PSET wasn't
       mapped already.  */
    GOMP_MAP_TO_PSET =			(GOMP_MAP_FLAG_SPECIAL_0 | 1),
    /* Must already be present.  */
    GOMP_MAP_FORCE_PRESENT =		(GOMP_MAP_FLAG_SPECIAL_0 | 2),
    /* Deallocate a mapping, without copying from device.  */
    GOMP_MAP_DELETE =			(GOMP_MAP_FLAG_SPECIAL_0 | 3),
    /* Is a device pointer.  OMP_CLAUSE_SIZE for these is unused; is implicitly
       POINTER_SIZE_UNITS.  */
    GOMP_MAP_FORCE_DEVICEPTR =		(GOMP_MAP_FLAG_SPECIAL_1 | 0),
    /* Do not map, copy bits for firstprivate instead.  */
    /* OpenACC device_resident.  */
    GOMP_MAP_DEVICE_RESIDENT =		(GOMP_MAP_FLAG_SPECIAL_1 | 1),
    /* OpenACC link.  */
    GOMP_MAP_LINK =			(GOMP_MAP_FLAG_SPECIAL_1 | 2),
    /* Allocate.  */
    GOMP_MAP_FIRSTPRIVATE =		(GOMP_MAP_FLAG_SPECIAL | 0),
    /* Similarly, but store the value in the pointer rather than
       pointed by the pointer.  */
    GOMP_MAP_FIRSTPRIVATE_INT =		(GOMP_MAP_FLAG_SPECIAL | 1),
    /* Pointer translate host address into device address and copy that
       back to host.  */
    GOMP_MAP_USE_DEVICE_PTR =		(GOMP_MAP_FLAG_SPECIAL | 2),
    /* Allocate a zero length array section.  Prefer next non-zero length
       mapping over previous non-zero length mapping over zero length mapping
       at the address.  If not already mapped, do nothing (and pointer translate
       to NULL).  */
    GOMP_MAP_ZERO_LEN_ARRAY_SECTION = 	(GOMP_MAP_FLAG_SPECIAL | 3),
    /* Allocate.  */
    GOMP_MAP_FORCE_ALLOC =		(GOMP_MAP_FLAG_FORCE | GOMP_MAP_ALLOC),
    /* ..., and copy to device.  */
    GOMP_MAP_FORCE_TO =			(GOMP_MAP_FLAG_FORCE | GOMP_MAP_TO),
    /* ..., and copy from device.  */
    GOMP_MAP_FORCE_FROM =		(GOMP_MAP_FLAG_FORCE | GOMP_MAP_FROM),
    /* ..., and copy to and from device.  */
    GOMP_MAP_FORCE_TOFROM =		(GOMP_MAP_FLAG_FORCE | GOMP_MAP_TOFROM),
    /* If not already present, allocate.  And unconditionally copy to
       device.  */
    GOMP_MAP_ALWAYS_TO =		(GOMP_MAP_FLAG_SPECIAL_2 | GOMP_MAP_TO),
    /* If not already present, allocate.  And unconditionally copy from
       device.  */
    GOMP_MAP_ALWAYS_FROM =		(GOMP_MAP_FLAG_SPECIAL_2
					 | GOMP_MAP_FROM),
    /* If not already present, allocate.  And unconditionally copy to and from
       device.  */
    GOMP_MAP_ALWAYS_TOFROM =		(GOMP_MAP_FLAG_SPECIAL_2
					 | GOMP_MAP_TOFROM),
    /* Map a sparse struct; the address is the base of the structure, alignment
       it's required alignment, and size is the number of adjacent entries
       that belong to the struct.  The adjacent entries should be sorted by
       increasing address, so it is easy to determine lowest needed address
       (address of the first adjacent entry) and highest needed address
       (address of the last adjacent entry plus its size).  */
    GOMP_MAP_STRUCT =			(GOMP_MAP_FLAG_SPECIAL_2
					 | GOMP_MAP_FLAG_SPECIAL | 0),
    /* On a location of a pointer/reference that is assumed to be already mapped
       earlier, store the translated address of the preceeding mapping.
       No refcount is bumped by this, and the store is done unconditionally.  */
    GOMP_MAP_ALWAYS_POINTER =		(GOMP_MAP_FLAG_SPECIAL_2
					 | GOMP_MAP_FLAG_SPECIAL | 1),
    /* Forced deallocation of zero length array section.  */
    GOMP_MAP_DELETE_ZERO_LEN_ARRAY_SECTION
      =					(GOMP_MAP_FLAG_SPECIAL_2
					 | GOMP_MAP_FLAG_SPECIAL | 3),
    /* Decrement usage count and deallocate if zero.  */
    GOMP_MAP_RELEASE =			(GOMP_MAP_FLAG_SPECIAL_2
					 | GOMP_MAP_DELETE),

    /* Internal to GCC, not used in libgomp.  */
    /* Do not map, but pointer assign a pointer instead.  */
    GOMP_MAP_FIRSTPRIVATE_POINTER =	(GOMP_MAP_LAST | 1),
    /* Do not map, but pointer assign a reference instead.  */
    GOMP_MAP_FIRSTPRIVATE_REFERENCE =	(GOMP_MAP_LAST | 2)
  };


int main(int argc,char** argv) {
  std::cout << "enumerator :: GOMP_MAP_ALLOC = " << gomp_map_kind::GOMP_MAP_ALLOC << std::endl;
  std::cout << "enumerator :: GOMP_MAP_TO = " << gomp_map_kind::GOMP_MAP_TO << std::endl;
  std::cout << "enumerator :: GOMP_MAP_FROM = " << gomp_map_kind::GOMP_MAP_FROM << std::endl;
  std::cout << "enumerator :: GOMP_MAP_TOFROM = " << gomp_map_kind::GOMP_MAP_TOFROM << std::endl;
  std::cout << "enumerator :: GOMP_MAP_POINTER = " << gomp_map_kind::GOMP_MAP_POINTER << std::endl;
  std::cout << "enumerator :: GOMP_MAP_POINTER = " << gomp_map_kind::GOMP_MAP_POINTER << std::endl;
  std::cout << "enumerator :: GOMP_MAP_TO_PSET = " << gomp_map_kind::GOMP_MAP_TO_PSET << std::endl;
  std::cout << "enumerator :: GOMP_MAP_FORCE_PRESENT = " << gomp_map_kind::GOMP_MAP_FORCE_PRESENT << std::endl;
  std::cout << "enumerator :: GOMP_MAP_DELETE = " << gomp_map_kind::GOMP_MAP_DELETE << std::endl;
  std::cout << "enumerator :: GOMP_MAP_FORCE_DEVICEPTR = " << gomp_map_kind::GOMP_MAP_FORCE_DEVICEPTR << std::endl;
  std::cout << "enumerator :: GOMP_MAP_DEVICE_RESIDENT = " << gomp_map_kind::GOMP_MAP_DEVICE_RESIDENT << std::endl;
  std::cout << "enumerator :: GOMP_MAP_LINK = " << gomp_map_kind::GOMP_MAP_LINK << std::endl;
  std::cout << "enumerator :: GOMP_MAP_FIRSTPRIVATE = " << gomp_map_kind::GOMP_MAP_FIRSTPRIVATE << std::endl;
  std::cout << "enumerator :: GOMP_MAP_FIRSTPRIVATE_INT = " << gomp_map_kind::GOMP_MAP_FIRSTPRIVATE_INT << std::endl;
  std::cout << "enumerator :: GOMP_MAP_USE_DEVICE_PTR = " << gomp_map_kind::GOMP_MAP_USE_DEVICE_PTR << std::endl;
  std::cout << "enumerator :: GOMP_MAP_ZERO_LEN_ARRAY_SECTION = " << gomp_map_kind::GOMP_MAP_ZERO_LEN_ARRAY_SECTION << std::endl;
  std::cout << "enumerator :: GOMP_MAP_FORCE_ALLOC = " << gomp_map_kind::GOMP_MAP_FORCE_ALLOC << std::endl;
  std::cout << "enumerator :: GOMP_MAP_FORCE_TO = " << gomp_map_kind::GOMP_MAP_FORCE_TO << std::endl;
  std::cout << "enumerator :: GOMP_MAP_FORCE_FROM = " << gomp_map_kind::GOMP_MAP_FORCE_FROM << std::endl;
  std::cout << "enumerator :: GOMP_MAP_FORCE_TOFROM = " << gomp_map_kind::GOMP_MAP_FORCE_TOFROM << std::endl;
  std::cout << "enumerator :: GOMP_MAP_ALWAYS_TO = " << gomp_map_kind::GOMP_MAP_ALWAYS_TO << std::endl;
  std::cout << "enumerator :: GOMP_MAP_ALWAYS_FROM = " << gomp_map_kind::GOMP_MAP_ALWAYS_FROM << std::endl;
  std::cout << "enumerator :: GOMP_MAP_ALWAYS_TOFROM = " << gomp_map_kind::GOMP_MAP_ALWAYS_TOFROM << std::endl;
  std::cout << "enumerator :: GOMP_MAP_STRUCT = " << gomp_map_kind::GOMP_MAP_STRUCT << std::endl;
  std::cout << "enumerator :: GOMP_MAP_ALWAYS_POINTER = " << gomp_map_kind::GOMP_MAP_ALWAYS_POINTER << std::endl;
  std::cout << "enumerator :: GOMP_MAP_DELETE_ZERO_LEN_ARRAY_SECTION = " << gomp_map_kind::GOMP_MAP_DELETE_ZERO_LEN_ARRAY_SECTION << std::endl;
  std::cout << "enumerator :: GOMP_MAP_RELEASE = " << gomp_map_kind::GOMP_MAP_RELEASE << std::endl;
  std::cout << "enumerator :: GOMP_MAP_FIRSTPRIVATE_POINTER = " << gomp_map_kind::GOMP_MAP_FIRSTPRIVATE_POINTER << std::endl;
  std::cout << "enumerator :: GOMP_MAP_FIRSTPRIVATE_REFERENCE = " << gomp_map_kind::GOMP_MAP_FIRSTPRIVATE_REFERENCE << std::endl;
  return 0;
}