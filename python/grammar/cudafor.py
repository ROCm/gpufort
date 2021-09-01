# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
#!/usr/bin/env python3
# NOTE: Everything relevant for the host is prefixed with HOST_
from grammar.cuda_enums import *
from grammar.cuda_libs import *

R_ARITH_OPERATOR=["**"]
L_ARITH_OPERATOR="+ - * /".split(" ")
COMP_OPERATOR_LOWER="<= >= == /= < > .eq. .ne. .lt. .gt. .le. .ge. .and. .or. .xor. .and. .or. .not. .eqv. .neqv.".split(" ") # can be uppercase too!

F77_KEYWORDS="""assign
backspace
block data
call
close
common
continue
data
dimension
do
else
else if
end
endfile
endif
entry
equivalence
external
format
type
function
goto
if
implicit
inquire
intrinsic
open
parameter
pause
print
program
read
return
rewind
rewrite
save
stop
subroutine
then
write""".split("\n")

F90_KEYWORDS="""allocatable
allocate
case
contains
cycle
deallocate
elsewhere
exit?
include
interface
intent
module
namelist
nullify
only
operator
optional
pointer
private
procedure
public
recursive
result
select
sequence
target
use
while
where""".split("\n") + F77_KEYWORDS

F95_KEYWORDS="""elemental
forall
pure""".split("\n") + F90_KEYWORDS

F03_KEYWORDS="""abstract
associate
asynchronous
bind
class
deferred
enum
enumerator
extends
final
flush
generic
import
non_overridable
nopass
pass
protected
value
volatile
wait""".split("\n") + F95_KEYWORDS

F08_KEYWORDS="""block
codimension
do concurrent
contiguous
critical
error stop
submodule
sync all
sync images
sync memory
lock
unlock""".split("\n") + F03_KEYWORDS

FORTRAN_VARIABLE_QUALIFIERS="allocatable pointer private public external parameter".split(" ")

FORTRAN_INTRINSICS="""abs
aimag
aint
anint
ceiling
cmplx
conjg
dim
floor
int
logical
max
min
mod
modulo
nint
real
sign
acos
acosh
asin
asinh
atan
atanh
atan2
bessel_j0
bessel_j1
bessel_jn
bessel_y0
bessel_y1
bessel_yn
cos
cosh
erf
erfc
exp
gamma
hypot
log
log10
log_gamma
sin
sinh
sqrt
tan
tanh
bit_size
digits
epsilon
huge
maxexponent
minexponent
precision
radix
range
selected_int_kind
selected_real_kind
tiny
btest
iand
ibclr
ibits
ibset
ieor
ior
ishft
ishftc
leadz
mvbits
not
popcnt
poppar
all
any
count
maxloc
maxval
minloc
minval
product
sum""".split("\n")
FORTRAN_INTRINSICS += ["ubound","lbound"]
FORTRAN_INTRINSICS += ["amax1","amin1","float","nint"]

# Host modules
HOST_MODULES="""cudafor
cublas
cufft
cusparse""".split("\n")

# Device Management
HOST_DEVICE_MANAGEMENT="""cudaChooseDevice
cudaDeviceGetAttribute
cudaDeviceGetCacheConfig
cudaDeviceGetLimit
cudaDeviceGetSharedMemConfig
cudaDeviceGetStreamPriorityRange
cudaDeviceReset
cudaDeviceSetCacheConfig
cudaDeviceSetLimit
cudaDeviceSetSharedMemConfig
cudaDeviceSynchronize
cudaGetDevice
cudaGetDeviceCount
cudaGetDeviceProperties
cudaSetDevice
cudaSetDeviceFlags
cudaSetValidDevices""".split("\n")

# Thread Management
HOST_THREAD_MANAGEMENT="""cudaThreadExit
cudaThreadSynchronize""".split("\n")

# Error Handling
HOST_ERROR_HANDLING="""cudaGetErrorString
cudaGetLastError
cudaPeekAtLastError""".split("\n")

# Stream Management
HOST_STREAM_MANAGEMENT="""cudaforGetDefaultStream
cudaforSetDefaultStream
cudaStreamAttachMemAsync
cudaStreamCreate
cudaStreamCreateWithFlags
cudaStreamCreateWithPriority
cudaStreamDestroy
cudaStreamGetPriority
cudaStreamQuery
cudaStreamSynchronize
cudaStreamWaitEvent""".split("\n")

# Event Management
HOST_EVENT_MANAGEMENT="""cudaEventCreate
cudaEventCreateWithFlags
cudaEventDestroy
cudaEventElapsedTime
cudaEventQuery
cudaEventRecord
cudaEventSynchronize""".split("\n")

# Execution Control
HOST_EXECUTION_CONTROL="""cudaFuncGetAttributes
cudaFuncSetAttribute
cudaFuncSetCacheConfig
cudaFuncSetSharedMemConfig
cudaSetDoubleForDevice
cudaSetDoubleForHost""".split("\n")

# Occupancy
HOST_OCCUPANCY="""cudaOccupancyMaxActiveBlocksPerMultiprocessor
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags""".split("\n")

# Memory Management
HOST_MEMORY_MANAGEMENT="""cudaFree
cudaFreeArray
cudaFreeHost
cudaGetSymbolAddress
cudaGetSymbolSize
cudaHostAlloc
cudaHostGetDevicePointer
cudaHostGetFlags
cudaHostRegister
cudaHostUnregister
cudaMalloc
cudaMallocArray
cudaMallocManaged
cudaMallocPitch
cudaMalloc3D
cudaMalloc3DArray
cudaMemAdvise
cudaMemcpy
cudaMemcpyArrayToArray
cudaMemcpyAsync
cudaMemcpyFromArray
cudaMemcpyFromSymbol
cudaMemcpyFromSymbolAsync
cudaMemcpyPeer
cudaMemcpyPeerAsync
cudaMemcpyToArray
cudaMemcpyToSymbol
cudaMemcpyToSymbolAsync
cudaMemcpy2D
cudaMemcpy2DArrayToArray
cudaMemcpy2DAsync
cudaMemcpy2DFromArray
cudaMemcpy2DToArray
cudaMemcpy3D
cudaMemcpy3DAsync
cudaMemGetInfo
cudaMemPrefetchAsync
cudaMemset
cudaMemsetAsync
cudaMemset2D
cudaMemset3D""".split("\n")

# Unified Addressing and Peer Device Memory Access
HOST_UNIFIED_ADDRESSING_AND_PEER_DEVICE_MEMORY_ACCESS="""cudaDeviceCanAccessPeer
cudaDeviceDisablePeerAccess
cudaDeviceEnablePeerAccess
cudaPointerGetAttributes""".split("\n")

# Version Management
HOST_VERSION_MANAGEMENT="""cudaDriverGetVersion
cudaRuntimeGetVersion""".split("\n")

ALL_HOST_ROUTINES=HOST_DEVICE_MANAGEMENT + HOST_THREAD_MANAGEMENT + HOST_ERROR_HANDLING + HOST_STREAM_MANAGEMENT + HOST_EVENT_MANAGEMENT + HOST_EXECUTION_CONTROL + HOST_OCCUPANCY + HOST_MEMORY_MANAGEMENT + HOST_UNIFIED_ADDRESSING_AND_PEER_DEVICE_MEMORY_ACCESS + HOST_VERSION_MANAGEMENT + FORTRAN_INTRINSICS

FUNCTION_QUALIFIERS="""host
global
device
host,device
grid_global""".split("\n")

DEVICE_VARIABLE_QUALIFIERS="""device
managed
constant
shared
pinned
texture""".split("\n")

DEVICE_PREDEFINED_VARIABLES="""threadidx
blockdim
blockidx
griddim
threadidx%x
threadidx%y
threadidx%z
blockdim%x
blockdim%y
blockdim%z
blockidx%x
blockidx%y
blockidx%z
griddim%x
griddim%y
griddim%z
warpsize""".split("\n")

# Synchronisation functions
DEVICE_SYNCHRONISATION_FUNCTIONS="""syncthreads
syncthreads_count
syncthreads_and
syncthread_or
syncwarp
threadfence
threadfence_block
threadfence_system""".split("\n")

# Atomics
DEVICE_ATOMICS="""atomicadd
atomicsub
atomicmax
atomicmin
atomicand
atomicor
atomicxor
atomicexch
atomicinc
atomicdec
atomiccas""".split("\n")

# Warp-Vote operations
DEVICE_WARP_VOTE_OPERATIONS="""allthreads
anythread
ballot
activemask
all_sync
any_sync
ballot_sync
match_all_sync
match_any_sync""".split("\n")

# Shuffle functions
DEVICE_SHUFFLE_FUNCTIONS="""__shfl
__shfl_up
__shfl_down
__shfl_xor""".split("\n")

# Device routines
DEVICE_ROUTINES="""__brev
__brevll
clock
clock64
__clz
__clzll
__cosf
cospi
cospif
__dadd_rd
__dadd_rn
__dadd_ru
__dadd_rz
__ddiv_rd
__ddiv_rn
__ddiv_ru
__ddiv_rz
__dmul_rd
__dmul_rn
__dmul_ru
__dmul_rz
__double2float_rd
__double2float_rn
__double2float_ru
__double2float_rz
__double2hiint
__double2int_rd
__double2int_rn
__double2int_ru
__double2int_rz
__double2loint
__double2ll_rd
__double2ll_rn
__double2ll_ru
__double2ll_rz
__double2uint_rd
__double2uint_rn
__double2uint_ru
__double2uint_rz
__double2ull_rd
__double2ull_rn
__double2ull_ru
__double2ull_rz
__double_as_longlong
__drcp_rd
__drcp_rn
__drcp_ru
__drcp_rz
__dsqrt_rd
__dsqrt_rn
__dsqrt_ru
__dsqrt_rz
__exp10f
__expf
__fadd_rd
__fadd_rn
__fadd_ru
__fadd_rz
__fdiv_rd
__fdiv_rn
__fdiv_ru
__fdiv_rz
fdivide
fdividef
__fdividef
__ffs
__ffsll
__float2half_rn
__float2int_rd
__float2int_rn
__float2int_ru
__float2int_rz
__float2ll_rd
__float2ll_rn
__float2ll_ru
__float2ll_rz
__float_as_int
__fma_rd
__fma_rn
__fma_ru
__fma_rz
__fmaf_rd
__fmaf_rn
__fmaf_ru
__fmaf_rz
__fmul_rd
__fmul_rn
__fmul_ru
__fmul_rz
__frcp_rd
__frcp_rn
__frcp_ru
__frcp_rz
__fsqrt_rd
__fsqrt_rn
__fsqrt_ru
__fsqrt_rz
__half2float
__hiloint2double
__int2double_rn
__int2float_rd
__int2float_rn
__int2float_ru
__int2float_rz
__int_as_float
__ll2double_rd
__ll2double_rn
__ll2double_ru
__ll2double_rz
__ll2float_rd
__ll2float_rn
__ll2float_ru
__ll2float_rz
__log10f
__log2f
__logf
__longlong_as_double
__mul24
__mulhi
__popc
__popcll
__powf
__sad
__saturatef
sincos
sincosf
sincospi
sincospif
__sinf
sinpi
sinpif
__tanf
__uint2double_rn
__uint2float_rd
__uint2float_rn
__uint2float_ru
__uint2float_rz
__ull2double_rd
__ull2double_rn
__ull2double_ru
__ull2double_rz
__ull2float_rd
__ull2float_rn
__ull2float_ru
__ull2float_rz
__umul24
__umulhi
__usa""".split("\n")

# libm routines
LIBM_ROUTINES="""cbrt
cbrtf
ceil
ceilf
copysign
copysign
expm1
expm1f
exp10
exp10f
exp2
exp2f
llround
llroundf
lrint
lrintf
lround
lroundf
logb
logbf
log1p
log1pf
log2
log2f
fabs
fabsf
floor
floorf
fma
fmaf
fmax
fmaxf
fmin
fminf
frexp
frexpf
ilogb
ilogbf
ldexp
ldexpf
llrint
llrintf
modf
modff
nearbyint
nearby
nextafter
nextaft
remainder
remai
remquo
remquof
rint
rintf
scalbn
scalbnf
scalbln
scalblnf
trunc
truncf""".split("\n")

ALL_DEVICE_ROUTINES = \
   DEVICE_SYNCHRONISATION_FUNCTIONS +\
   DEVICE_ATOMICS +\
   DEVICE_WARP_VOTE_OPERATIONS +\
   DEVICE_SHUFFLE_FUNCTIONS +\
   DEVICE_ROUTINES + \
   FORTRAN_INTRINSICS +\
   LIBM_ROUTINES

CUDA_FORTRAN_KEYWORDS=F08_KEYWORDS+DEVICE_VARIABLE_QUALIFIERS+FUNCTION_QUALIFIERS
CUDA_FORTRAN_VARIABLE_QUALIFIERS=FORTRAN_VARIABLE_QUALIFIERS + DEVICE_VARIABLE_QUALIFIERS

R_ARITH_OPERATOR_STR=" ".join(R_ARITH_OPERATOR)
L_ARITH_OPERATOR_STR=" ".join(L_ARITH_OPERATOR)
COMP_OPERATOR_LOWER_STR=" ".join(COMP_OPERATOR_LOWER)
F77_KEYWORDS_STR=" ".join(F77_KEYWORDS)
F90_KEYWORDS_STR=" ".join(F90_KEYWORDS)
F95_KEYWORDS_STR=" ".join(F95_KEYWORDS)
F03_KEYWORDS_STR=" ".join(F03_KEYWORDS)
F08_KEYWORDS_STR=" ".join(F08_KEYWORDS)
FORTRAN_VARIABLE_QUALIFIERS_STR=" ".join(FORTRAN_VARIABLE_QUALIFIERS)
CUDA_FORTRAN_KEYWORDS_STR=" ".join(CUDA_FORTRAN_KEYWORDS)
CUDA_FORTRAN_VARIABLE_QUALIFIERS_STR=" ".join(CUDA_FORTRAN_VARIABLE_QUALIFIERS)
HOST_MODULES_STR=" ".join(HOST_MODULES)
HOST_DEVICE_MANAGEMENT_STR=" ".join(HOST_DEVICE_MANAGEMENT)
HOST_THREAD_MANAGEMENT_STR=" ".join(HOST_THREAD_MANAGEMENT)
HOST_ERROR_HANDLING_STR=" ".join(HOST_ERROR_HANDLING)
HOST_STREAM_MANAGEMENT_STR=" ".join(HOST_STREAM_MANAGEMENT)
HOST_EVENT_MANAGEMENT_STR=" ".join(HOST_EVENT_MANAGEMENT)
HOST_EXECUTION_CONTROL_STR=" ".join(HOST_EXECUTION_CONTROL)
HOST_OCCUPANCY_STR=" ".join(HOST_OCCUPANCY)
HOST_MEMORY_MANAGEMENT_STR=" ".join(HOST_MEMORY_MANAGEMENT)
HOST_UNIFIED_ADDRESSING_AND_PEER_DEVICE_MEMORY_ACCESS_STR=" ".join(HOST_UNIFIED_ADDRESSING_AND_PEER_DEVICE_MEMORY_ACCESS)
HOST_VERSION_MANAGEMENT_STR=" ".join(HOST_VERSION_MANAGEMENT)
ALL_HOST_ROUTINES_STR=" ".join(ALL_HOST_ROUTINES)
FUNCTION_QUALIFIERS_STR=" ".join(FUNCTION_QUALIFIERS)
DEVICE_VARIABLE_QUALIFIERS_STR=" ".join(DEVICE_VARIABLE_QUALIFIERS)
DEVICE_PREDEFINED_VARIABLES_STR=" ".join(DEVICE_PREDEFINED_VARIABLES)
FORTRAN_INTRINSICS_STR=" ".join(FORTRAN_INTRINSICS)
DEVICE_SYNCHRONISATION_FUNCTIONS_STR=" ".join(DEVICE_SYNCHRONISATION_FUNCTIONS)
DEVICE_ATOMICS_STR=" ".join(DEVICE_ATOMICS)
DEVICE_WARP_VOTE_OPERATIONS_STR=" ".join(DEVICE_WARP_VOTE_OPERATIONS)
DEVICE_SHUFFLE_FUNCTIONS_STR=" ".join(DEVICE_SHUFFLE_FUNCTIONS)
DEVICE_ROUTINES_STR=" ".join(DEVICE_ROUTINES)
ALL_DEVICE_ROUTINES_STR=" ".join(ALL_DEVICE_ROUTINES)
LIBM_ROUTINES_STR=" ".join(LIBM_ROUTINES)
