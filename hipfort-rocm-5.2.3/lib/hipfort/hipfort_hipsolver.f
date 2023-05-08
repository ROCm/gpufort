!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! ==============================================================================
! hipfort: FORTRAN Interfaces for GPU kernels
! ==============================================================================
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
! [MITx11 License]
! 
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
! 
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
! 
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
! THE SOFTWARE.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
          
           
module hipfort_hipsolver
  use hipfort_hipsolver_enums
  implicit none

 
  
  interface hipsolverCreate
#ifdef USE_CUDA_NAMES
    function hipsolverCreate_(handle) bind(c, name="cusolverCreate")
#else
    function hipsolverCreate_(handle) bind(c, name="hipsolverCreate")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCreate_
      type(c_ptr) :: handle
    end function

  end interface
  
  interface hipsolverDestroy
#ifdef USE_CUDA_NAMES
    function hipsolverDestroy_(handle) bind(c, name="cusolverDestroy")
#else
    function hipsolverDestroy_(handle) bind(c, name="hipsolverDestroy")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDestroy_
      type(c_ptr),value :: handle
    end function

  end interface
  
  interface hipsolverSetStream
#ifdef USE_CUDA_NAMES
    function hipsolverSetStream_(handle,streamId) bind(c, name="cusolverSetStream")
#else
    function hipsolverSetStream_(handle,streamId) bind(c, name="hipsolverSetStream")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSetStream_
      type(c_ptr),value :: handle
      type(c_ptr),value :: streamId
    end function

  end interface
  
  interface hipsolverGetStream
#ifdef USE_CUDA_NAMES
    function hipsolverGetStream_(handle,streamId) bind(c, name="cusolverGetStream")
#else
    function hipsolverGetStream_(handle,streamId) bind(c, name="hipsolverGetStream")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverGetStream_
      type(c_ptr),value :: handle
      type(c_ptr) :: streamId
    end function

  end interface
  
  interface hipsolverSorgbr_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverSorgbr_bufferSize_(handle,side,m,n,k,A,lda,tau,lwork) bind(c, name="cusolverSorgbr_bufferSize")
#else
    function hipsolverSorgbr_bufferSize_(handle,side,m,n,k,A,lda,tau,lwork) bind(c, name="hipsolverSorgbr_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgbr_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)),value :: side
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int),value :: k
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      real(c_float) :: tau
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSorgbr_bufferSize_full_rank,&
      hipsolverSorgbr_bufferSize_rank_0,&
      hipsolverSorgbr_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverDorgbr_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverDorgbr_bufferSize_(handle,side,m,n,k,A,lda,tau,lwork) bind(c, name="cusolverDorgbr_bufferSize")
#else
    function hipsolverDorgbr_bufferSize_(handle,side,m,n,k,A,lda,tau,lwork) bind(c, name="hipsolverDorgbr_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgbr_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)),value :: side
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int),value :: k
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      real(c_double) :: tau
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDorgbr_bufferSize_full_rank,&
      hipsolverDorgbr_bufferSize_rank_0,&
      hipsolverDorgbr_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverCungbr_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverCungbr_bufferSize_(handle,side,m,n,k,A,lda,tau,lwork) bind(c, name="cusolverCungbr_bufferSize")
#else
    function hipsolverCungbr_bufferSize_(handle,side,m,n,k,A,lda,tau,lwork) bind(c, name="hipsolverCungbr_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungbr_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)),value :: side
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int),value :: k
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      complex(c_float_complex) :: tau
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCungbr_bufferSize_full_rank,&
      hipsolverCungbr_bufferSize_rank_0,&
      hipsolverCungbr_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverZungbr_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverZungbr_bufferSize_(handle,side,m,n,k,A,lda,tau,lwork) bind(c, name="cusolverZungbr_bufferSize")
#else
    function hipsolverZungbr_bufferSize_(handle,side,m,n,k,A,lda,tau,lwork) bind(c, name="hipsolverZungbr_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungbr_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)),value :: side
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int),value :: k
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      complex(c_double_complex) :: tau
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZungbr_bufferSize_full_rank,&
      hipsolverZungbr_bufferSize_rank_0,&
      hipsolverZungbr_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverSorgbr
#ifdef USE_CUDA_NAMES
    function hipsolverSorgbr_(handle,side,m,n,k,A,lda,tau,work,lwork,devInfo) bind(c, name="cusolverSorgbr")
#else
    function hipsolverSorgbr_(handle,side,m,n,k,A,lda,tau,work,lwork,devInfo) bind(c, name="hipsolverSorgbr")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgbr_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)),value :: side
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int),value :: k
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      real(c_float) :: tau
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSorgbr_full_rank,&
      hipsolverSorgbr_rank_0,&
      hipsolverSorgbr_rank_1
#endif
  end interface
  
  interface hipsolverDorgbr
#ifdef USE_CUDA_NAMES
    function hipsolverDorgbr_(handle,side,m,n,k,A,lda,tau,work,lwork,devInfo) bind(c, name="cusolverDorgbr")
#else
    function hipsolverDorgbr_(handle,side,m,n,k,A,lda,tau,work,lwork,devInfo) bind(c, name="hipsolverDorgbr")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgbr_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)),value :: side
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int),value :: k
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      real(c_double) :: tau
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDorgbr_full_rank,&
      hipsolverDorgbr_rank_0,&
      hipsolverDorgbr_rank_1
#endif
  end interface
  
  interface hipsolverCungbr
#ifdef USE_CUDA_NAMES
    function hipsolverCungbr_(handle,side,m,n,k,A,lda,tau,work,lwork,devInfo) bind(c, name="cusolverCungbr")
#else
    function hipsolverCungbr_(handle,side,m,n,k,A,lda,tau,work,lwork,devInfo) bind(c, name="hipsolverCungbr")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungbr_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)),value :: side
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int),value :: k
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      complex(c_float_complex) :: tau
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCungbr_full_rank,&
      hipsolverCungbr_rank_0,&
      hipsolverCungbr_rank_1
#endif
  end interface
  
  interface hipsolverZungbr
#ifdef USE_CUDA_NAMES
    function hipsolverZungbr_(handle,side,m,n,k,A,lda,tau,work,lwork,devInfo) bind(c, name="cusolverZungbr")
#else
    function hipsolverZungbr_(handle,side,m,n,k,A,lda,tau,work,lwork,devInfo) bind(c, name="hipsolverZungbr")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungbr_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)),value :: side
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int),value :: k
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      complex(c_double_complex) :: tau
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZungbr_full_rank,&
      hipsolverZungbr_rank_0,&
      hipsolverZungbr_rank_1
#endif
  end interface
  
  interface hipsolverSorgqr_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverSorgqr_bufferSize_(handle,m,n,k,A,lda,tau,lwork) bind(c, name="cusolverSorgqr_bufferSize")
#else
    function hipsolverSorgqr_bufferSize_(handle,m,n,k,A,lda,tau,lwork) bind(c, name="hipsolverSorgqr_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgqr_bufferSize_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int),value :: k
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      real(c_float) :: tau
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSorgqr_bufferSize_full_rank,&
      hipsolverSorgqr_bufferSize_rank_0,&
      hipsolverSorgqr_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverDorgqr_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverDorgqr_bufferSize_(handle,m,n,k,A,lda,tau,lwork) bind(c, name="cusolverDorgqr_bufferSize")
#else
    function hipsolverDorgqr_bufferSize_(handle,m,n,k,A,lda,tau,lwork) bind(c, name="hipsolverDorgqr_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgqr_bufferSize_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int),value :: k
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      real(c_double) :: tau
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDorgqr_bufferSize_full_rank,&
      hipsolverDorgqr_bufferSize_rank_0,&
      hipsolverDorgqr_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverCungqr_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverCungqr_bufferSize_(handle,m,n,k,A,lda,tau,lwork) bind(c, name="cusolverCungqr_bufferSize")
#else
    function hipsolverCungqr_bufferSize_(handle,m,n,k,A,lda,tau,lwork) bind(c, name="hipsolverCungqr_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungqr_bufferSize_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int),value :: k
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      complex(c_float_complex) :: tau
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCungqr_bufferSize_full_rank,&
      hipsolverCungqr_bufferSize_rank_0,&
      hipsolverCungqr_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverZungqr_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverZungqr_bufferSize_(handle,m,n,k,A,lda,tau,lwork) bind(c, name="cusolverZungqr_bufferSize")
#else
    function hipsolverZungqr_bufferSize_(handle,m,n,k,A,lda,tau,lwork) bind(c, name="hipsolverZungqr_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungqr_bufferSize_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int),value :: k
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      complex(c_double_complex) :: tau
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZungqr_bufferSize_full_rank,&
      hipsolverZungqr_bufferSize_rank_0,&
      hipsolverZungqr_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverSorgqr
#ifdef USE_CUDA_NAMES
    function hipsolverSorgqr_(handle,m,n,k,A,lda,tau,work,lwork,devInfo) bind(c, name="cusolverSorgqr")
#else
    function hipsolverSorgqr_(handle,m,n,k,A,lda,tau,work,lwork,devInfo) bind(c, name="hipsolverSorgqr")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgqr_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int),value :: k
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      real(c_float) :: tau
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSorgqr_full_rank,&
      hipsolverSorgqr_rank_0,&
      hipsolverSorgqr_rank_1
#endif
  end interface
  
  interface hipsolverDorgqr
#ifdef USE_CUDA_NAMES
    function hipsolverDorgqr_(handle,m,n,k,A,lda,tau,work,lwork,devInfo) bind(c, name="cusolverDorgqr")
#else
    function hipsolverDorgqr_(handle,m,n,k,A,lda,tau,work,lwork,devInfo) bind(c, name="hipsolverDorgqr")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgqr_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int),value :: k
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      real(c_double) :: tau
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDorgqr_full_rank,&
      hipsolverDorgqr_rank_0,&
      hipsolverDorgqr_rank_1
#endif
  end interface
  
  interface hipsolverCungqr
#ifdef USE_CUDA_NAMES
    function hipsolverCungqr_(handle,m,n,k,A,lda,tau,work,lwork,devInfo) bind(c, name="cusolverCungqr")
#else
    function hipsolverCungqr_(handle,m,n,k,A,lda,tau,work,lwork,devInfo) bind(c, name="hipsolverCungqr")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungqr_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int),value :: k
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      complex(c_float_complex) :: tau
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCungqr_full_rank,&
      hipsolverCungqr_rank_0,&
      hipsolverCungqr_rank_1
#endif
  end interface
  
  interface hipsolverZungqr
#ifdef USE_CUDA_NAMES
    function hipsolverZungqr_(handle,m,n,k,A,lda,tau,work,lwork,devInfo) bind(c, name="cusolverZungqr")
#else
    function hipsolverZungqr_(handle,m,n,k,A,lda,tau,work,lwork,devInfo) bind(c, name="hipsolverZungqr")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungqr_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int),value :: k
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      complex(c_double_complex) :: tau
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZungqr_full_rank,&
      hipsolverZungqr_rank_0,&
      hipsolverZungqr_rank_1
#endif
  end interface
  
  interface hipsolverSorgtr_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverSorgtr_bufferSize_(handle,uplo,n,A,lda,tau,lwork) bind(c, name="cusolverSorgtr_bufferSize")
#else
    function hipsolverSorgtr_bufferSize_(handle,uplo,n,A,lda,tau,lwork) bind(c, name="hipsolverSorgtr_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgtr_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      real(c_float) :: tau
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSorgtr_bufferSize_full_rank,&
      hipsolverSorgtr_bufferSize_rank_0,&
      hipsolverSorgtr_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverDorgtr_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverDorgtr_bufferSize_(handle,uplo,n,A,lda,tau,lwork) bind(c, name="cusolverDorgtr_bufferSize")
#else
    function hipsolverDorgtr_bufferSize_(handle,uplo,n,A,lda,tau,lwork) bind(c, name="hipsolverDorgtr_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgtr_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      real(c_double) :: tau
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDorgtr_bufferSize_full_rank,&
      hipsolverDorgtr_bufferSize_rank_0,&
      hipsolverDorgtr_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverCungtr_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverCungtr_bufferSize_(handle,uplo,n,A,lda,tau,lwork) bind(c, name="cusolverCungtr_bufferSize")
#else
    function hipsolverCungtr_bufferSize_(handle,uplo,n,A,lda,tau,lwork) bind(c, name="hipsolverCungtr_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungtr_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      complex(c_float_complex) :: tau
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCungtr_bufferSize_full_rank,&
      hipsolverCungtr_bufferSize_rank_0,&
      hipsolverCungtr_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverZungtr_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverZungtr_bufferSize_(handle,uplo,n,A,lda,tau,lwork) bind(c, name="cusolverZungtr_bufferSize")
#else
    function hipsolverZungtr_bufferSize_(handle,uplo,n,A,lda,tau,lwork) bind(c, name="hipsolverZungtr_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungtr_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      complex(c_double_complex) :: tau
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZungtr_bufferSize_full_rank,&
      hipsolverZungtr_bufferSize_rank_0,&
      hipsolverZungtr_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverSorgtr
#ifdef USE_CUDA_NAMES
    function hipsolverSorgtr_(handle,uplo,n,A,lda,tau,work,lwork,devInfo) bind(c, name="cusolverSorgtr")
#else
    function hipsolverSorgtr_(handle,uplo,n,A,lda,tau,work,lwork,devInfo) bind(c, name="hipsolverSorgtr")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgtr_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      real(c_float) :: tau
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSorgtr_full_rank,&
      hipsolverSorgtr_rank_0,&
      hipsolverSorgtr_rank_1
#endif
  end interface
  
  interface hipsolverDorgtr
#ifdef USE_CUDA_NAMES
    function hipsolverDorgtr_(handle,uplo,n,A,lda,tau,work,lwork,devInfo) bind(c, name="cusolverDorgtr")
#else
    function hipsolverDorgtr_(handle,uplo,n,A,lda,tau,work,lwork,devInfo) bind(c, name="hipsolverDorgtr")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgtr_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      real(c_double) :: tau
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDorgtr_full_rank,&
      hipsolverDorgtr_rank_0,&
      hipsolverDorgtr_rank_1
#endif
  end interface
  
  interface hipsolverCungtr
#ifdef USE_CUDA_NAMES
    function hipsolverCungtr_(handle,uplo,n,A,lda,tau,work,lwork,devInfo) bind(c, name="cusolverCungtr")
#else
    function hipsolverCungtr_(handle,uplo,n,A,lda,tau,work,lwork,devInfo) bind(c, name="hipsolverCungtr")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungtr_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      complex(c_float_complex) :: tau
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCungtr_full_rank,&
      hipsolverCungtr_rank_0,&
      hipsolverCungtr_rank_1
#endif
  end interface
  
  interface hipsolverZungtr
#ifdef USE_CUDA_NAMES
    function hipsolverZungtr_(handle,uplo,n,A,lda,tau,work,lwork,devInfo) bind(c, name="cusolverZungtr")
#else
    function hipsolverZungtr_(handle,uplo,n,A,lda,tau,work,lwork,devInfo) bind(c, name="hipsolverZungtr")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungtr_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      complex(c_double_complex) :: tau
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZungtr_full_rank,&
      hipsolverZungtr_rank_0,&
      hipsolverZungtr_rank_1
#endif
  end interface
  
  interface hipsolverSormqr_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverSormqr_bufferSize_(handle,side,trans,m,n,k,A,lda,tau,C,ldc,lwork) bind(c, name="cusolverSormqr_bufferSize")
#else
    function hipsolverSormqr_bufferSize_(handle,side,trans,m,n,k,A,lda,tau,C,ldc,lwork) bind(c, name="hipsolverSormqr_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSormqr_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)),value :: side
      integer(kind(HIPSOLVER_OP_N)),value :: trans
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int),value :: k
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      real(c_float) :: tau
      type(c_ptr),value :: C
      integer(c_int),value :: ldc
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSormqr_bufferSize_full_rank,&
      hipsolverSormqr_bufferSize_rank_0,&
      hipsolverSormqr_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverDormqr_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverDormqr_bufferSize_(handle,side,trans,m,n,k,A,lda,tau,C,ldc,lwork) bind(c, name="cusolverDormqr_bufferSize")
#else
    function hipsolverDormqr_bufferSize_(handle,side,trans,m,n,k,A,lda,tau,C,ldc,lwork) bind(c, name="hipsolverDormqr_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDormqr_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)),value :: side
      integer(kind(HIPSOLVER_OP_N)),value :: trans
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int),value :: k
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      real(c_double) :: tau
      type(c_ptr),value :: C
      integer(c_int),value :: ldc
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDormqr_bufferSize_full_rank,&
      hipsolverDormqr_bufferSize_rank_0,&
      hipsolverDormqr_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverCunmqr_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverCunmqr_bufferSize_(handle,side,trans,m,n,k,A,lda,tau,C,ldc,lwork) bind(c, name="cusolverCunmqr_bufferSize")
#else
    function hipsolverCunmqr_bufferSize_(handle,side,trans,m,n,k,A,lda,tau,C,ldc,lwork) bind(c, name="hipsolverCunmqr_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCunmqr_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)),value :: side
      integer(kind(HIPSOLVER_OP_N)),value :: trans
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int),value :: k
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      complex(c_float_complex) :: tau
      type(c_ptr),value :: C
      integer(c_int),value :: ldc
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCunmqr_bufferSize_full_rank,&
      hipsolverCunmqr_bufferSize_rank_0,&
      hipsolverCunmqr_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverZunmqr_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverZunmqr_bufferSize_(handle,side,trans,m,n,k,A,lda,tau,C,ldc,lwork) bind(c, name="cusolverZunmqr_bufferSize")
#else
    function hipsolverZunmqr_bufferSize_(handle,side,trans,m,n,k,A,lda,tau,C,ldc,lwork) bind(c, name="hipsolverZunmqr_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZunmqr_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)),value :: side
      integer(kind(HIPSOLVER_OP_N)),value :: trans
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int),value :: k
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      complex(c_double_complex) :: tau
      type(c_ptr),value :: C
      integer(c_int),value :: ldc
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZunmqr_bufferSize_full_rank,&
      hipsolverZunmqr_bufferSize_rank_0,&
      hipsolverZunmqr_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverSormqr
#ifdef USE_CUDA_NAMES
    function hipsolverSormqr_(handle,side,trans,m,n,k,A,lda,tau,C,ldc,work,lwork,devInfo) bind(c, name="cusolverSormqr")
#else
    function hipsolverSormqr_(handle,side,trans,m,n,k,A,lda,tau,C,ldc,work,lwork,devInfo) bind(c, name="hipsolverSormqr")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSormqr_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)),value :: side
      integer(kind(HIPSOLVER_OP_N)),value :: trans
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int),value :: k
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      real(c_float) :: tau
      type(c_ptr),value :: C
      integer(c_int),value :: ldc
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSormqr_full_rank,&
      hipsolverSormqr_rank_0,&
      hipsolverSormqr_rank_1
#endif
  end interface
  
  interface hipsolverDormqr
#ifdef USE_CUDA_NAMES
    function hipsolverDormqr_(handle,side,trans,m,n,k,A,lda,tau,C,ldc,work,lwork,devInfo) bind(c, name="cusolverDormqr")
#else
    function hipsolverDormqr_(handle,side,trans,m,n,k,A,lda,tau,C,ldc,work,lwork,devInfo) bind(c, name="hipsolverDormqr")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDormqr_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)),value :: side
      integer(kind(HIPSOLVER_OP_N)),value :: trans
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int),value :: k
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      real(c_double) :: tau
      type(c_ptr),value :: C
      integer(c_int),value :: ldc
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDormqr_full_rank,&
      hipsolverDormqr_rank_0,&
      hipsolverDormqr_rank_1
#endif
  end interface
  
  interface hipsolverCunmqr
#ifdef USE_CUDA_NAMES
    function hipsolverCunmqr_(handle,side,trans,m,n,k,A,lda,tau,C,ldc,work,lwork,devInfo) bind(c, name="cusolverCunmqr")
#else
    function hipsolverCunmqr_(handle,side,trans,m,n,k,A,lda,tau,C,ldc,work,lwork,devInfo) bind(c, name="hipsolverCunmqr")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCunmqr_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)),value :: side
      integer(kind(HIPSOLVER_OP_N)),value :: trans
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int),value :: k
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      complex(c_float_complex) :: tau
      type(c_ptr),value :: C
      integer(c_int),value :: ldc
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCunmqr_full_rank,&
      hipsolverCunmqr_rank_0,&
      hipsolverCunmqr_rank_1
#endif
  end interface
  
  interface hipsolverZunmqr
#ifdef USE_CUDA_NAMES
    function hipsolverZunmqr_(handle,side,trans,m,n,k,A,lda,tau,C,ldc,work,lwork,devInfo) bind(c, name="cusolverZunmqr")
#else
    function hipsolverZunmqr_(handle,side,trans,m,n,k,A,lda,tau,C,ldc,work,lwork,devInfo) bind(c, name="hipsolverZunmqr")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZunmqr_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)),value :: side
      integer(kind(HIPSOLVER_OP_N)),value :: trans
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int),value :: k
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      complex(c_double_complex) :: tau
      type(c_ptr),value :: C
      integer(c_int),value :: ldc
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZunmqr_full_rank,&
      hipsolverZunmqr_rank_0,&
      hipsolverZunmqr_rank_1
#endif
  end interface
  
  interface hipsolverSormtr_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverSormtr_bufferSize_(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,lwork) bind(c, name="cusolverSormtr_bufferSize")
#else
    function hipsolverSormtr_bufferSize_(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,lwork) bind(c, name="hipsolverSormtr_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSormtr_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)),value :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(kind(HIPSOLVER_OP_N)),value :: trans
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      real(c_float) :: tau
      type(c_ptr),value :: C
      integer(c_int),value :: ldc
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSormtr_bufferSize_full_rank,&
      hipsolverSormtr_bufferSize_rank_0,&
      hipsolverSormtr_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverDormtr_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverDormtr_bufferSize_(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,lwork) bind(c, name="cusolverDormtr_bufferSize")
#else
    function hipsolverDormtr_bufferSize_(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,lwork) bind(c, name="hipsolverDormtr_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDormtr_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)),value :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(kind(HIPSOLVER_OP_N)),value :: trans
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      real(c_double) :: tau
      type(c_ptr),value :: C
      integer(c_int),value :: ldc
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDormtr_bufferSize_full_rank,&
      hipsolverDormtr_bufferSize_rank_0,&
      hipsolverDormtr_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverCunmtr_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverCunmtr_bufferSize_(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,lwork) bind(c, name="cusolverCunmtr_bufferSize")
#else
    function hipsolverCunmtr_bufferSize_(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,lwork) bind(c, name="hipsolverCunmtr_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCunmtr_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)),value :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(kind(HIPSOLVER_OP_N)),value :: trans
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      complex(c_float_complex) :: tau
      type(c_ptr),value :: C
      integer(c_int),value :: ldc
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCunmtr_bufferSize_full_rank,&
      hipsolverCunmtr_bufferSize_rank_0,&
      hipsolverCunmtr_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverZunmtr_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverZunmtr_bufferSize_(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,lwork) bind(c, name="cusolverZunmtr_bufferSize")
#else
    function hipsolverZunmtr_bufferSize_(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,lwork) bind(c, name="hipsolverZunmtr_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZunmtr_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)),value :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(kind(HIPSOLVER_OP_N)),value :: trans
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      complex(c_double_complex) :: tau
      type(c_ptr),value :: C
      integer(c_int),value :: ldc
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZunmtr_bufferSize_full_rank,&
      hipsolverZunmtr_bufferSize_rank_0,&
      hipsolverZunmtr_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverSormtr
#ifdef USE_CUDA_NAMES
    function hipsolverSormtr_(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,work,lwork,devInfo) bind(c, name="cusolverSormtr")
#else
    function hipsolverSormtr_(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,work,lwork,devInfo) bind(c, name="hipsolverSormtr")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSormtr_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)),value :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(kind(HIPSOLVER_OP_N)),value :: trans
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      real(c_float) :: tau
      type(c_ptr),value :: C
      integer(c_int),value :: ldc
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSormtr_full_rank,&
      hipsolverSormtr_rank_0,&
      hipsolverSormtr_rank_1
#endif
  end interface
  
  interface hipsolverDormtr
#ifdef USE_CUDA_NAMES
    function hipsolverDormtr_(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,work,lwork,devInfo) bind(c, name="cusolverDormtr")
#else
    function hipsolverDormtr_(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,work,lwork,devInfo) bind(c, name="hipsolverDormtr")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDormtr_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)),value :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(kind(HIPSOLVER_OP_N)),value :: trans
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      real(c_double) :: tau
      type(c_ptr),value :: C
      integer(c_int),value :: ldc
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDormtr_full_rank,&
      hipsolverDormtr_rank_0,&
      hipsolverDormtr_rank_1
#endif
  end interface
  
  interface hipsolverCunmtr
#ifdef USE_CUDA_NAMES
    function hipsolverCunmtr_(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,work,lwork,devInfo) bind(c, name="cusolverCunmtr")
#else
    function hipsolverCunmtr_(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,work,lwork,devInfo) bind(c, name="hipsolverCunmtr")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCunmtr_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)),value :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(kind(HIPSOLVER_OP_N)),value :: trans
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      complex(c_float_complex) :: tau
      type(c_ptr),value :: C
      integer(c_int),value :: ldc
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCunmtr_full_rank,&
      hipsolverCunmtr_rank_0,&
      hipsolverCunmtr_rank_1
#endif
  end interface
  
  interface hipsolverZunmtr
#ifdef USE_CUDA_NAMES
    function hipsolverZunmtr_(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,work,lwork,devInfo) bind(c, name="cusolverZunmtr")
#else
    function hipsolverZunmtr_(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,work,lwork,devInfo) bind(c, name="hipsolverZunmtr")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZunmtr_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)),value :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(kind(HIPSOLVER_OP_N)),value :: trans
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      complex(c_double_complex) :: tau
      type(c_ptr),value :: C
      integer(c_int),value :: ldc
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZunmtr_full_rank,&
      hipsolverZunmtr_rank_0,&
      hipsolverZunmtr_rank_1
#endif
  end interface
  
  interface hipsolverSgebrd_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverSgebrd_bufferSize_(handle,m,n,lwork) bind(c, name="cusolverSgebrd_bufferSize")
#else
    function hipsolverSgebrd_bufferSize_(handle,m,n,lwork) bind(c, name="hipsolverSgebrd_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgebrd_bufferSize_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int) :: lwork
    end function

  end interface
  
  interface hipsolverDgebrd_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverDgebrd_bufferSize_(handle,m,n,lwork) bind(c, name="cusolverDgebrd_bufferSize")
#else
    function hipsolverDgebrd_bufferSize_(handle,m,n,lwork) bind(c, name="hipsolverDgebrd_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgebrd_bufferSize_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int) :: lwork
    end function

  end interface
  
  interface hipsolverCgebrd_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverCgebrd_bufferSize_(handle,m,n,lwork) bind(c, name="cusolverCgebrd_bufferSize")
#else
    function hipsolverCgebrd_bufferSize_(handle,m,n,lwork) bind(c, name="hipsolverCgebrd_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgebrd_bufferSize_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int) :: lwork
    end function

  end interface
  
  interface hipsolverZgebrd_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverZgebrd_bufferSize_(handle,m,n,lwork) bind(c, name="cusolverZgebrd_bufferSize")
#else
    function hipsolverZgebrd_bufferSize_(handle,m,n,lwork) bind(c, name="hipsolverZgebrd_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgebrd_bufferSize_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      integer(c_int) :: lwork
    end function

  end interface
  
  interface hipsolverSgebrd
#ifdef USE_CUDA_NAMES
    function hipsolverSgebrd_(handle,m,n,A,lda,D,E,tauq,taup,work,lwork,devInfo) bind(c, name="cusolverSgebrd")
#else
    function hipsolverSgebrd_(handle,m,n,A,lda,D,E,tauq,taup,work,lwork,devInfo) bind(c, name="hipsolverSgebrd")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgebrd_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: D
      type(c_ptr),value :: E
      type(c_ptr),value :: tauq
      type(c_ptr),value :: taup
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSgebrd_full_rank,&
      hipsolverSgebrd_rank_0,&
      hipsolverSgebrd_rank_1
#endif
  end interface
  
  interface hipsolverDgebrd
#ifdef USE_CUDA_NAMES
    function hipsolverDgebrd_(handle,m,n,A,lda,D,E,tauq,taup,work,lwork,devInfo) bind(c, name="cusolverDgebrd")
#else
    function hipsolverDgebrd_(handle,m,n,A,lda,D,E,tauq,taup,work,lwork,devInfo) bind(c, name="hipsolverDgebrd")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgebrd_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: D
      type(c_ptr),value :: E
      type(c_ptr),value :: tauq
      type(c_ptr),value :: taup
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDgebrd_full_rank,&
      hipsolverDgebrd_rank_0,&
      hipsolverDgebrd_rank_1
#endif
  end interface
  
  interface hipsolverCgebrd
#ifdef USE_CUDA_NAMES
    function hipsolverCgebrd_(handle,m,n,A,lda,D,E,tauq,taup,work,lwork,devInfo) bind(c, name="cusolverCgebrd")
#else
    function hipsolverCgebrd_(handle,m,n,A,lda,D,E,tauq,taup,work,lwork,devInfo) bind(c, name="hipsolverCgebrd")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgebrd_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: D
      type(c_ptr),value :: E
      type(c_ptr),value :: tauq
      type(c_ptr),value :: taup
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCgebrd_full_rank,&
      hipsolverCgebrd_rank_0,&
      hipsolverCgebrd_rank_1
#endif
  end interface
  
  interface hipsolverZgebrd
#ifdef USE_CUDA_NAMES
    function hipsolverZgebrd_(handle,m,n,A,lda,D,E,tauq,taup,work,lwork,devInfo) bind(c, name="cusolverZgebrd")
#else
    function hipsolverZgebrd_(handle,m,n,A,lda,D,E,tauq,taup,work,lwork,devInfo) bind(c, name="hipsolverZgebrd")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgebrd_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: D
      type(c_ptr),value :: E
      type(c_ptr),value :: tauq
      type(c_ptr),value :: taup
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZgebrd_full_rank,&
      hipsolverZgebrd_rank_0,&
      hipsolverZgebrd_rank_1
#endif
  end interface
  
  interface hipsolverSgeqrf_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverSgeqrf_bufferSize_(handle,m,n,A,lda,lwork) bind(c, name="cusolverSgeqrf_bufferSize")
#else
    function hipsolverSgeqrf_bufferSize_(handle,m,n,A,lda,lwork) bind(c, name="hipsolverSgeqrf_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgeqrf_bufferSize_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSgeqrf_bufferSize_full_rank,&
      hipsolverSgeqrf_bufferSize_rank_0,&
      hipsolverSgeqrf_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverDgeqrf_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverDgeqrf_bufferSize_(handle,m,n,A,lda,lwork) bind(c, name="cusolverDgeqrf_bufferSize")
#else
    function hipsolverDgeqrf_bufferSize_(handle,m,n,A,lda,lwork) bind(c, name="hipsolverDgeqrf_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgeqrf_bufferSize_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDgeqrf_bufferSize_full_rank,&
      hipsolverDgeqrf_bufferSize_rank_0,&
      hipsolverDgeqrf_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverCgeqrf_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverCgeqrf_bufferSize_(handle,m,n,A,lda,lwork) bind(c, name="cusolverCgeqrf_bufferSize")
#else
    function hipsolverCgeqrf_bufferSize_(handle,m,n,A,lda,lwork) bind(c, name="hipsolverCgeqrf_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgeqrf_bufferSize_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCgeqrf_bufferSize_full_rank,&
      hipsolverCgeqrf_bufferSize_rank_0,&
      hipsolverCgeqrf_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverZgeqrf_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverZgeqrf_bufferSize_(handle,m,n,A,lda,lwork) bind(c, name="cusolverZgeqrf_bufferSize")
#else
    function hipsolverZgeqrf_bufferSize_(handle,m,n,A,lda,lwork) bind(c, name="hipsolverZgeqrf_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgeqrf_bufferSize_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZgeqrf_bufferSize_full_rank,&
      hipsolverZgeqrf_bufferSize_rank_0,&
      hipsolverZgeqrf_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverSgeqrf
#ifdef USE_CUDA_NAMES
    function hipsolverSgeqrf_(handle,m,n,A,lda,tau,work,lwork,devInfo) bind(c, name="cusolverSgeqrf")
#else
    function hipsolverSgeqrf_(handle,m,n,A,lda,tau,work,lwork,devInfo) bind(c, name="hipsolverSgeqrf")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgeqrf_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      real(c_float) :: tau
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSgeqrf_full_rank,&
      hipsolverSgeqrf_rank_0,&
      hipsolverSgeqrf_rank_1
#endif
  end interface
  
  interface hipsolverDgeqrf
#ifdef USE_CUDA_NAMES
    function hipsolverDgeqrf_(handle,m,n,A,lda,tau,work,lwork,devInfo) bind(c, name="cusolverDgeqrf")
#else
    function hipsolverDgeqrf_(handle,m,n,A,lda,tau,work,lwork,devInfo) bind(c, name="hipsolverDgeqrf")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgeqrf_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      real(c_double) :: tau
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDgeqrf_full_rank,&
      hipsolverDgeqrf_rank_0,&
      hipsolverDgeqrf_rank_1
#endif
  end interface
  
  interface hipsolverCgeqrf
#ifdef USE_CUDA_NAMES
    function hipsolverCgeqrf_(handle,m,n,A,lda,tau,work,lwork,devInfo) bind(c, name="cusolverCgeqrf")
#else
    function hipsolverCgeqrf_(handle,m,n,A,lda,tau,work,lwork,devInfo) bind(c, name="hipsolverCgeqrf")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgeqrf_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      complex(c_float_complex) :: tau
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCgeqrf_full_rank,&
      hipsolverCgeqrf_rank_0,&
      hipsolverCgeqrf_rank_1
#endif
  end interface
  
  interface hipsolverZgeqrf
#ifdef USE_CUDA_NAMES
    function hipsolverZgeqrf_(handle,m,n,A,lda,tau,work,lwork,devInfo) bind(c, name="cusolverZgeqrf")
#else
    function hipsolverZgeqrf_(handle,m,n,A,lda,tau,work,lwork,devInfo) bind(c, name="hipsolverZgeqrf")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgeqrf_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      complex(c_double_complex) :: tau
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZgeqrf_full_rank,&
      hipsolverZgeqrf_rank_0,&
      hipsolverZgeqrf_rank_1
#endif
  end interface
  
  interface hipsolverSSgesv_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverSSgesv_bufferSize_(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,lwork) bind(c, name="cusolverSSgesv_bufferSize")
#else
    function hipsolverSSgesv_bufferSize_(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,lwork) bind(c, name="hipsolverSSgesv_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSSgesv_bufferSize_
      type(c_ptr),value :: handle
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: devIpiv
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: X
      integer(c_int),value :: ldx
      integer(c_size_t) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSSgesv_bufferSize_full_rank,&
      hipsolverSSgesv_bufferSize_rank_0,&
      hipsolverSSgesv_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverDDgesv_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverDDgesv_bufferSize_(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,lwork) bind(c, name="cusolverDDgesv_bufferSize")
#else
    function hipsolverDDgesv_bufferSize_(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,lwork) bind(c, name="hipsolverDDgesv_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDDgesv_bufferSize_
      type(c_ptr),value :: handle
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: devIpiv
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: X
      integer(c_int),value :: ldx
      integer(c_size_t) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDDgesv_bufferSize_full_rank,&
      hipsolverDDgesv_bufferSize_rank_0,&
      hipsolverDDgesv_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverCCgesv_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverCCgesv_bufferSize_(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,lwork) bind(c, name="cusolverCCgesv_bufferSize")
#else
    function hipsolverCCgesv_bufferSize_(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,lwork) bind(c, name="hipsolverCCgesv_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCCgesv_bufferSize_
      type(c_ptr),value :: handle
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: devIpiv
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: X
      integer(c_int),value :: ldx
      integer(c_size_t) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCCgesv_bufferSize_full_rank,&
      hipsolverCCgesv_bufferSize_rank_0,&
      hipsolverCCgesv_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverZZgesv_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverZZgesv_bufferSize_(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,lwork) bind(c, name="cusolverZZgesv_bufferSize")
#else
    function hipsolverZZgesv_bufferSize_(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,lwork) bind(c, name="hipsolverZZgesv_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZZgesv_bufferSize_
      type(c_ptr),value :: handle
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: devIpiv
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: X
      integer(c_int),value :: ldx
      integer(c_size_t) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZZgesv_bufferSize_full_rank,&
      hipsolverZZgesv_bufferSize_rank_0,&
      hipsolverZZgesv_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverSSgesv
#ifdef USE_CUDA_NAMES
    function hipsolverSSgesv_(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,work,lwork,niters,devInfo) bind(c, name="cusolverSSgesv")
#else
    function hipsolverSSgesv_(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,work,lwork,niters,devInfo) bind(c, name="hipsolverSSgesv")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSSgesv_
      type(c_ptr),value :: handle
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: devIpiv
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: X
      integer(c_int),value :: ldx
      type(c_ptr),value :: work
      integer(c_size_t),value :: lwork
      type(c_ptr),value :: niters
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSSgesv_full_rank,&
      hipsolverSSgesv_rank_0,&
      hipsolverSSgesv_rank_1
#endif
  end interface
  
  interface hipsolverDDgesv
#ifdef USE_CUDA_NAMES
    function hipsolverDDgesv_(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,work,lwork,niters,devInfo) bind(c, name="cusolverDDgesv")
#else
    function hipsolverDDgesv_(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,work,lwork,niters,devInfo) bind(c, name="hipsolverDDgesv")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDDgesv_
      type(c_ptr),value :: handle
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: devIpiv
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: X
      integer(c_int),value :: ldx
      type(c_ptr),value :: work
      integer(c_size_t),value :: lwork
      type(c_ptr),value :: niters
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDDgesv_full_rank,&
      hipsolverDDgesv_rank_0,&
      hipsolverDDgesv_rank_1
#endif
  end interface
  
  interface hipsolverCCgesv
#ifdef USE_CUDA_NAMES
    function hipsolverCCgesv_(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,work,lwork,niters,devInfo) bind(c, name="cusolverCCgesv")
#else
    function hipsolverCCgesv_(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,work,lwork,niters,devInfo) bind(c, name="hipsolverCCgesv")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCCgesv_
      type(c_ptr),value :: handle
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: devIpiv
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: X
      integer(c_int),value :: ldx
      type(c_ptr),value :: work
      integer(c_size_t),value :: lwork
      type(c_ptr),value :: niters
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCCgesv_full_rank,&
      hipsolverCCgesv_rank_0,&
      hipsolverCCgesv_rank_1
#endif
  end interface
  
  interface hipsolverZZgesv
#ifdef USE_CUDA_NAMES
    function hipsolverZZgesv_(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,work,lwork,niters,devInfo) bind(c, name="cusolverZZgesv")
#else
    function hipsolverZZgesv_(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,work,lwork,niters,devInfo) bind(c, name="hipsolverZZgesv")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZZgesv_
      type(c_ptr),value :: handle
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: devIpiv
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: X
      integer(c_int),value :: ldx
      type(c_ptr),value :: work
      integer(c_size_t),value :: lwork
      type(c_ptr),value :: niters
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZZgesv_full_rank,&
      hipsolverZZgesv_rank_0,&
      hipsolverZZgesv_rank_1
#endif
  end interface
  
  interface hipsolverSgetrf_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverSgetrf_bufferSize_(handle,m,n,A,lda,lwork) bind(c, name="cusolverSgetrf_bufferSize")
#else
    function hipsolverSgetrf_bufferSize_(handle,m,n,A,lda,lwork) bind(c, name="hipsolverSgetrf_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgetrf_bufferSize_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSgetrf_bufferSize_full_rank,&
      hipsolverSgetrf_bufferSize_rank_0,&
      hipsolverSgetrf_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverDgetrf_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverDgetrf_bufferSize_(handle,m,n,A,lda,lwork) bind(c, name="cusolverDgetrf_bufferSize")
#else
    function hipsolverDgetrf_bufferSize_(handle,m,n,A,lda,lwork) bind(c, name="hipsolverDgetrf_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgetrf_bufferSize_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDgetrf_bufferSize_full_rank,&
      hipsolverDgetrf_bufferSize_rank_0,&
      hipsolverDgetrf_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverCgetrf_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverCgetrf_bufferSize_(handle,m,n,A,lda,lwork) bind(c, name="cusolverCgetrf_bufferSize")
#else
    function hipsolverCgetrf_bufferSize_(handle,m,n,A,lda,lwork) bind(c, name="hipsolverCgetrf_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgetrf_bufferSize_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCgetrf_bufferSize_full_rank,&
      hipsolverCgetrf_bufferSize_rank_0,&
      hipsolverCgetrf_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverZgetrf_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverZgetrf_bufferSize_(handle,m,n,A,lda,lwork) bind(c, name="cusolverZgetrf_bufferSize")
#else
    function hipsolverZgetrf_bufferSize_(handle,m,n,A,lda,lwork) bind(c, name="hipsolverZgetrf_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgetrf_bufferSize_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZgetrf_bufferSize_full_rank,&
      hipsolverZgetrf_bufferSize_rank_0,&
      hipsolverZgetrf_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverSgetrf
#ifdef USE_CUDA_NAMES
    function hipsolverSgetrf_(handle,m,n,A,lda,work,lwork,devIpiv,devInfo) bind(c, name="cusolverSgetrf")
#else
    function hipsolverSgetrf_(handle,m,n,A,lda,work,lwork,devIpiv,devInfo) bind(c, name="hipsolverSgetrf")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgetrf_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      type(c_ptr),value :: devIpiv
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSgetrf_full_rank,&
      hipsolverSgetrf_rank_0,&
      hipsolverSgetrf_rank_1
#endif
  end interface
  
  interface hipsolverDgetrf
#ifdef USE_CUDA_NAMES
    function hipsolverDgetrf_(handle,m,n,A,lda,work,lwork,devIpiv,devInfo) bind(c, name="cusolverDgetrf")
#else
    function hipsolverDgetrf_(handle,m,n,A,lda,work,lwork,devIpiv,devInfo) bind(c, name="hipsolverDgetrf")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgetrf_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      type(c_ptr),value :: devIpiv
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDgetrf_full_rank,&
      hipsolverDgetrf_rank_0,&
      hipsolverDgetrf_rank_1
#endif
  end interface
  
  interface hipsolverCgetrf
#ifdef USE_CUDA_NAMES
    function hipsolverCgetrf_(handle,m,n,A,lda,work,lwork,devIpiv,devInfo) bind(c, name="cusolverCgetrf")
#else
    function hipsolverCgetrf_(handle,m,n,A,lda,work,lwork,devIpiv,devInfo) bind(c, name="hipsolverCgetrf")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgetrf_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      type(c_ptr),value :: devIpiv
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCgetrf_full_rank,&
      hipsolverCgetrf_rank_0,&
      hipsolverCgetrf_rank_1
#endif
  end interface
  
  interface hipsolverZgetrf
#ifdef USE_CUDA_NAMES
    function hipsolverZgetrf_(handle,m,n,A,lda,work,lwork,devIpiv,devInfo) bind(c, name="cusolverZgetrf")
#else
    function hipsolverZgetrf_(handle,m,n,A,lda,work,lwork,devIpiv,devInfo) bind(c, name="hipsolverZgetrf")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgetrf_
      type(c_ptr),value :: handle
      integer(c_int),value :: m
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      type(c_ptr),value :: devIpiv
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZgetrf_full_rank,&
      hipsolverZgetrf_rank_0,&
      hipsolverZgetrf_rank_1
#endif
  end interface
  
  interface hipsolverSgetrs_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverSgetrs_bufferSize_(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,lwork) bind(c, name="cusolverSgetrs_bufferSize")
#else
    function hipsolverSgetrs_bufferSize_(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,lwork) bind(c, name="hipsolverSgetrs_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgetrs_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_OP_N)),value :: trans
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: devIpiv
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSgetrs_bufferSize_full_rank,&
      hipsolverSgetrs_bufferSize_rank_0,&
      hipsolverSgetrs_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverDgetrs_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverDgetrs_bufferSize_(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,lwork) bind(c, name="cusolverDgetrs_bufferSize")
#else
    function hipsolverDgetrs_bufferSize_(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,lwork) bind(c, name="hipsolverDgetrs_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgetrs_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_OP_N)),value :: trans
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: devIpiv
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDgetrs_bufferSize_full_rank,&
      hipsolverDgetrs_bufferSize_rank_0,&
      hipsolverDgetrs_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverCgetrs_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverCgetrs_bufferSize_(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,lwork) bind(c, name="cusolverCgetrs_bufferSize")
#else
    function hipsolverCgetrs_bufferSize_(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,lwork) bind(c, name="hipsolverCgetrs_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgetrs_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_OP_N)),value :: trans
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: devIpiv
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCgetrs_bufferSize_full_rank,&
      hipsolverCgetrs_bufferSize_rank_0,&
      hipsolverCgetrs_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverZgetrs_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverZgetrs_bufferSize_(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,lwork) bind(c, name="cusolverZgetrs_bufferSize")
#else
    function hipsolverZgetrs_bufferSize_(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,lwork) bind(c, name="hipsolverZgetrs_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgetrs_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_OP_N)),value :: trans
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: devIpiv
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZgetrs_bufferSize_full_rank,&
      hipsolverZgetrs_bufferSize_rank_0,&
      hipsolverZgetrs_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverSgetrs
#ifdef USE_CUDA_NAMES
    function hipsolverSgetrs_(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,work,lwork,devInfo) bind(c, name="cusolverSgetrs")
#else
    function hipsolverSgetrs_(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,work,lwork,devInfo) bind(c, name="hipsolverSgetrs")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgetrs_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_OP_N)),value :: trans
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: devIpiv
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSgetrs_full_rank,&
      hipsolverSgetrs_rank_0,&
      hipsolverSgetrs_rank_1
#endif
  end interface
  
  interface hipsolverDgetrs
#ifdef USE_CUDA_NAMES
    function hipsolverDgetrs_(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,work,lwork,devInfo) bind(c, name="cusolverDgetrs")
#else
    function hipsolverDgetrs_(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,work,lwork,devInfo) bind(c, name="hipsolverDgetrs")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgetrs_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_OP_N)),value :: trans
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: devIpiv
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDgetrs_full_rank,&
      hipsolverDgetrs_rank_0,&
      hipsolverDgetrs_rank_1
#endif
  end interface
  
  interface hipsolverCgetrs
#ifdef USE_CUDA_NAMES
    function hipsolverCgetrs_(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,work,lwork,devInfo) bind(c, name="cusolverCgetrs")
#else
    function hipsolverCgetrs_(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,work,lwork,devInfo) bind(c, name="hipsolverCgetrs")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgetrs_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_OP_N)),value :: trans
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: devIpiv
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCgetrs_full_rank,&
      hipsolverCgetrs_rank_0,&
      hipsolverCgetrs_rank_1
#endif
  end interface
  
  interface hipsolverZgetrs
#ifdef USE_CUDA_NAMES
    function hipsolverZgetrs_(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,work,lwork,devInfo) bind(c, name="cusolverZgetrs")
#else
    function hipsolverZgetrs_(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,work,lwork,devInfo) bind(c, name="hipsolverZgetrs")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgetrs_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_OP_N)),value :: trans
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: devIpiv
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZgetrs_full_rank,&
      hipsolverZgetrs_rank_0,&
      hipsolverZgetrs_rank_1
#endif
  end interface
  
  interface hipsolverSpotrf_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverSpotrf_bufferSize_(handle,uplo,n,A,lda,lwork) bind(c, name="cusolverSpotrf_bufferSize")
#else
    function hipsolverSpotrf_bufferSize_(handle,uplo,n,A,lda,lwork) bind(c, name="hipsolverSpotrf_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrf_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSpotrf_bufferSize_full_rank,&
      hipsolverSpotrf_bufferSize_rank_0,&
      hipsolverSpotrf_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverDpotrf_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverDpotrf_bufferSize_(handle,uplo,n,A,lda,lwork) bind(c, name="cusolverDpotrf_bufferSize")
#else
    function hipsolverDpotrf_bufferSize_(handle,uplo,n,A,lda,lwork) bind(c, name="hipsolverDpotrf_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrf_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDpotrf_bufferSize_full_rank,&
      hipsolverDpotrf_bufferSize_rank_0,&
      hipsolverDpotrf_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverCpotrf_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverCpotrf_bufferSize_(handle,uplo,n,A,lda,lwork) bind(c, name="cusolverCpotrf_bufferSize")
#else
    function hipsolverCpotrf_bufferSize_(handle,uplo,n,A,lda,lwork) bind(c, name="hipsolverCpotrf_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrf_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCpotrf_bufferSize_full_rank,&
      hipsolverCpotrf_bufferSize_rank_0,&
      hipsolverCpotrf_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverZpotrf_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverZpotrf_bufferSize_(handle,uplo,n,A,lda,lwork) bind(c, name="cusolverZpotrf_bufferSize")
#else
    function hipsolverZpotrf_bufferSize_(handle,uplo,n,A,lda,lwork) bind(c, name="hipsolverZpotrf_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrf_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZpotrf_bufferSize_full_rank,&
      hipsolverZpotrf_bufferSize_rank_0,&
      hipsolverZpotrf_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverSpotrf
#ifdef USE_CUDA_NAMES
    function hipsolverSpotrf_(handle,uplo,n,A,lda,work,lwork,devInfo) bind(c, name="cusolverSpotrf")
#else
    function hipsolverSpotrf_(handle,uplo,n,A,lda,work,lwork,devInfo) bind(c, name="hipsolverSpotrf")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrf_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSpotrf_full_rank,&
      hipsolverSpotrf_rank_0,&
      hipsolverSpotrf_rank_1
#endif
  end interface
  
  interface hipsolverDpotrf
#ifdef USE_CUDA_NAMES
    function hipsolverDpotrf_(handle,uplo,n,A,lda,work,lwork,devInfo) bind(c, name="cusolverDpotrf")
#else
    function hipsolverDpotrf_(handle,uplo,n,A,lda,work,lwork,devInfo) bind(c, name="hipsolverDpotrf")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrf_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDpotrf_full_rank,&
      hipsolverDpotrf_rank_0,&
      hipsolverDpotrf_rank_1
#endif
  end interface
  
  interface hipsolverCpotrf
#ifdef USE_CUDA_NAMES
    function hipsolverCpotrf_(handle,uplo,n,A,lda,work,lwork,devInfo) bind(c, name="cusolverCpotrf")
#else
    function hipsolverCpotrf_(handle,uplo,n,A,lda,work,lwork,devInfo) bind(c, name="hipsolverCpotrf")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrf_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCpotrf_full_rank,&
      hipsolverCpotrf_rank_0,&
      hipsolverCpotrf_rank_1
#endif
  end interface
  
  interface hipsolverZpotrf
#ifdef USE_CUDA_NAMES
    function hipsolverZpotrf_(handle,uplo,n,A,lda,work,lwork,devInfo) bind(c, name="cusolverZpotrf")
#else
    function hipsolverZpotrf_(handle,uplo,n,A,lda,work,lwork,devInfo) bind(c, name="hipsolverZpotrf")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrf_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZpotrf_full_rank,&
      hipsolverZpotrf_rank_0,&
      hipsolverZpotrf_rank_1
#endif
  end interface
  
  interface hipsolverSpotrfBatched_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverSpotrfBatched_bufferSize_(handle,uplo,n,A,lda,lwork,batch_count) bind(c, name="cusolverSpotrfBatched_bufferSize")
#else
    function hipsolverSpotrfBatched_bufferSize_(handle,uplo,n,A,lda,lwork,batch_count) bind(c, name="hipsolverSpotrfBatched_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrfBatched_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr) :: A
      integer(c_int),value :: lda
      integer(c_int) :: lwork
      integer(c_int),value :: batch_count
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSpotrfBatched_bufferSize_full_rank,&
      hipsolverSpotrfBatched_bufferSize_rank_0,&
      hipsolverSpotrfBatched_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverDpotrfBatched_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverDpotrfBatched_bufferSize_(handle,uplo,n,A,lda,lwork,batch_count) bind(c, name="cusolverDpotrfBatched_bufferSize")
#else
    function hipsolverDpotrfBatched_bufferSize_(handle,uplo,n,A,lda,lwork,batch_count) bind(c, name="hipsolverDpotrfBatched_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrfBatched_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr) :: A
      integer(c_int),value :: lda
      integer(c_int) :: lwork
      integer(c_int),value :: batch_count
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDpotrfBatched_bufferSize_full_rank,&
      hipsolverDpotrfBatched_bufferSize_rank_0,&
      hipsolverDpotrfBatched_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverCpotrfBatched_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverCpotrfBatched_bufferSize_(handle,uplo,n,A,lda,lwork,batch_count) bind(c, name="cusolverCpotrfBatched_bufferSize")
#else
    function hipsolverCpotrfBatched_bufferSize_(handle,uplo,n,A,lda,lwork,batch_count) bind(c, name="hipsolverCpotrfBatched_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrfBatched_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr) :: A
      integer(c_int),value :: lda
      integer(c_int) :: lwork
      integer(c_int),value :: batch_count
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCpotrfBatched_bufferSize_full_rank,&
      hipsolverCpotrfBatched_bufferSize_rank_0,&
      hipsolverCpotrfBatched_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverZpotrfBatched_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverZpotrfBatched_bufferSize_(handle,uplo,n,A,lda,lwork,batch_count) bind(c, name="cusolverZpotrfBatched_bufferSize")
#else
    function hipsolverZpotrfBatched_bufferSize_(handle,uplo,n,A,lda,lwork,batch_count) bind(c, name="hipsolverZpotrfBatched_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrfBatched_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr) :: A
      integer(c_int),value :: lda
      integer(c_int) :: lwork
      integer(c_int),value :: batch_count
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZpotrfBatched_bufferSize_full_rank,&
      hipsolverZpotrfBatched_bufferSize_rank_0,&
      hipsolverZpotrfBatched_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverSpotrfBatched
#ifdef USE_CUDA_NAMES
    function hipsolverSpotrfBatched_(handle,uplo,n,A,lda,work,lwork,devInfo,batch_count) bind(c, name="cusolverSpotrfBatched")
#else
    function hipsolverSpotrfBatched_(handle,uplo,n,A,lda,work,lwork,devInfo,batch_count) bind(c, name="hipsolverSpotrfBatched")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrfBatched_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr) :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
      integer(c_int),value :: batch_count
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSpotrfBatched_full_rank,&
      hipsolverSpotrfBatched_rank_0,&
      hipsolverSpotrfBatched_rank_1
#endif
  end interface
  
  interface hipsolverDpotrfBatched
#ifdef USE_CUDA_NAMES
    function hipsolverDpotrfBatched_(handle,uplo,n,A,lda,work,lwork,devInfo,batch_count) bind(c, name="cusolverDpotrfBatched")
#else
    function hipsolverDpotrfBatched_(handle,uplo,n,A,lda,work,lwork,devInfo,batch_count) bind(c, name="hipsolverDpotrfBatched")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrfBatched_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr) :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
      integer(c_int),value :: batch_count
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDpotrfBatched_full_rank,&
      hipsolverDpotrfBatched_rank_0,&
      hipsolverDpotrfBatched_rank_1
#endif
  end interface
  
  interface hipsolverCpotrfBatched
#ifdef USE_CUDA_NAMES
    function hipsolverCpotrfBatched_(handle,uplo,n,A,lda,work,lwork,devInfo,batch_count) bind(c, name="cusolverCpotrfBatched")
#else
    function hipsolverCpotrfBatched_(handle,uplo,n,A,lda,work,lwork,devInfo,batch_count) bind(c, name="hipsolverCpotrfBatched")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrfBatched_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr) :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
      integer(c_int),value :: batch_count
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCpotrfBatched_full_rank,&
      hipsolverCpotrfBatched_rank_0,&
      hipsolverCpotrfBatched_rank_1
#endif
  end interface
  
  interface hipsolverZpotrfBatched
#ifdef USE_CUDA_NAMES
    function hipsolverZpotrfBatched_(handle,uplo,n,A,lda,work,lwork,devInfo,batch_count) bind(c, name="cusolverZpotrfBatched")
#else
    function hipsolverZpotrfBatched_(handle,uplo,n,A,lda,work,lwork,devInfo,batch_count) bind(c, name="hipsolverZpotrfBatched")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrfBatched_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr) :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
      integer(c_int),value :: batch_count
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZpotrfBatched_full_rank,&
      hipsolverZpotrfBatched_rank_0,&
      hipsolverZpotrfBatched_rank_1
#endif
  end interface
  
  interface hipsolverSpotri_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverSpotri_bufferSize_(handle,uplo,n,A,lda,lwork) bind(c, name="cusolverSpotri_bufferSize")
#else
    function hipsolverSpotri_bufferSize_(handle,uplo,n,A,lda,lwork) bind(c, name="hipsolverSpotri_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotri_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSpotri_bufferSize_full_rank,&
      hipsolverSpotri_bufferSize_rank_0,&
      hipsolverSpotri_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverDpotri_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverDpotri_bufferSize_(handle,uplo,n,A,lda,lwork) bind(c, name="cusolverDpotri_bufferSize")
#else
    function hipsolverDpotri_bufferSize_(handle,uplo,n,A,lda,lwork) bind(c, name="hipsolverDpotri_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotri_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDpotri_bufferSize_full_rank,&
      hipsolverDpotri_bufferSize_rank_0,&
      hipsolverDpotri_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverCpotri_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverCpotri_bufferSize_(handle,uplo,n,A,lda,lwork) bind(c, name="cusolverCpotri_bufferSize")
#else
    function hipsolverCpotri_bufferSize_(handle,uplo,n,A,lda,lwork) bind(c, name="hipsolverCpotri_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotri_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCpotri_bufferSize_full_rank,&
      hipsolverCpotri_bufferSize_rank_0,&
      hipsolverCpotri_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverZpotri_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverZpotri_bufferSize_(handle,uplo,n,A,lda,lwork) bind(c, name="cusolverZpotri_bufferSize")
#else
    function hipsolverZpotri_bufferSize_(handle,uplo,n,A,lda,lwork) bind(c, name="hipsolverZpotri_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotri_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZpotri_bufferSize_full_rank,&
      hipsolverZpotri_bufferSize_rank_0,&
      hipsolverZpotri_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverSpotri
#ifdef USE_CUDA_NAMES
    function hipsolverSpotri_(handle,uplo,n,A,lda,work,lwork,devInfo) bind(c, name="cusolverSpotri")
#else
    function hipsolverSpotri_(handle,uplo,n,A,lda,work,lwork,devInfo) bind(c, name="hipsolverSpotri")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotri_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSpotri_full_rank,&
      hipsolverSpotri_rank_0,&
      hipsolverSpotri_rank_1
#endif
  end interface
  
  interface hipsolverDpotri
#ifdef USE_CUDA_NAMES
    function hipsolverDpotri_(handle,uplo,n,A,lda,work,lwork,devInfo) bind(c, name="cusolverDpotri")
#else
    function hipsolverDpotri_(handle,uplo,n,A,lda,work,lwork,devInfo) bind(c, name="hipsolverDpotri")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotri_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDpotri_full_rank,&
      hipsolverDpotri_rank_0,&
      hipsolverDpotri_rank_1
#endif
  end interface
  
  interface hipsolverCpotri
#ifdef USE_CUDA_NAMES
    function hipsolverCpotri_(handle,uplo,n,A,lda,work,lwork,devInfo) bind(c, name="cusolverCpotri")
#else
    function hipsolverCpotri_(handle,uplo,n,A,lda,work,lwork,devInfo) bind(c, name="hipsolverCpotri")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotri_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCpotri_full_rank,&
      hipsolverCpotri_rank_0,&
      hipsolverCpotri_rank_1
#endif
  end interface
  
  interface hipsolverZpotri
#ifdef USE_CUDA_NAMES
    function hipsolverZpotri_(handle,uplo,n,A,lda,work,lwork,devInfo) bind(c, name="cusolverZpotri")
#else
    function hipsolverZpotri_(handle,uplo,n,A,lda,work,lwork,devInfo) bind(c, name="hipsolverZpotri")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotri_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZpotri_full_rank,&
      hipsolverZpotri_rank_0,&
      hipsolverZpotri_rank_1
#endif
  end interface
  
  interface hipsolverSpotrs_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverSpotrs_bufferSize_(handle,uplo,n,nrhs,A,lda,B,ldb,lwork) bind(c, name="cusolverSpotrs_bufferSize")
#else
    function hipsolverSpotrs_bufferSize_(handle,uplo,n,nrhs,A,lda,B,ldb,lwork) bind(c, name="hipsolverSpotrs_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrs_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSpotrs_bufferSize_full_rank,&
      hipsolverSpotrs_bufferSize_rank_0,&
      hipsolverSpotrs_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverDpotrs_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverDpotrs_bufferSize_(handle,uplo,n,nrhs,A,lda,B,ldb,lwork) bind(c, name="cusolverDpotrs_bufferSize")
#else
    function hipsolverDpotrs_bufferSize_(handle,uplo,n,nrhs,A,lda,B,ldb,lwork) bind(c, name="hipsolverDpotrs_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrs_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDpotrs_bufferSize_full_rank,&
      hipsolverDpotrs_bufferSize_rank_0,&
      hipsolverDpotrs_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverCpotrs_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverCpotrs_bufferSize_(handle,uplo,n,nrhs,A,lda,B,ldb,lwork) bind(c, name="cusolverCpotrs_bufferSize")
#else
    function hipsolverCpotrs_bufferSize_(handle,uplo,n,nrhs,A,lda,B,ldb,lwork) bind(c, name="hipsolverCpotrs_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrs_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCpotrs_bufferSize_full_rank,&
      hipsolverCpotrs_bufferSize_rank_0,&
      hipsolverCpotrs_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverZpotrs_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverZpotrs_bufferSize_(handle,uplo,n,nrhs,A,lda,B,ldb,lwork) bind(c, name="cusolverZpotrs_bufferSize")
#else
    function hipsolverZpotrs_bufferSize_(handle,uplo,n,nrhs,A,lda,B,ldb,lwork) bind(c, name="hipsolverZpotrs_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrs_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZpotrs_bufferSize_full_rank,&
      hipsolverZpotrs_bufferSize_rank_0,&
      hipsolverZpotrs_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverSpotrs
#ifdef USE_CUDA_NAMES
    function hipsolverSpotrs_(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo) bind(c, name="cusolverSpotrs")
#else
    function hipsolverSpotrs_(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo) bind(c, name="hipsolverSpotrs")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrs_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSpotrs_full_rank,&
      hipsolverSpotrs_rank_0,&
      hipsolverSpotrs_rank_1
#endif
  end interface
  
  interface hipsolverDpotrs
#ifdef USE_CUDA_NAMES
    function hipsolverDpotrs_(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo) bind(c, name="cusolverDpotrs")
#else
    function hipsolverDpotrs_(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo) bind(c, name="hipsolverDpotrs")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrs_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDpotrs_full_rank,&
      hipsolverDpotrs_rank_0,&
      hipsolverDpotrs_rank_1
#endif
  end interface
  
  interface hipsolverCpotrs
#ifdef USE_CUDA_NAMES
    function hipsolverCpotrs_(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo) bind(c, name="cusolverCpotrs")
#else
    function hipsolverCpotrs_(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo) bind(c, name="hipsolverCpotrs")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrs_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCpotrs_full_rank,&
      hipsolverCpotrs_rank_0,&
      hipsolverCpotrs_rank_1
#endif
  end interface
  
  interface hipsolverZpotrs
#ifdef USE_CUDA_NAMES
    function hipsolverZpotrs_(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo) bind(c, name="cusolverZpotrs")
#else
    function hipsolverZpotrs_(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo) bind(c, name="hipsolverZpotrs")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrs_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZpotrs_full_rank,&
      hipsolverZpotrs_rank_0,&
      hipsolverZpotrs_rank_1
#endif
  end interface
  
  interface hipsolverSpotrsBatched_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverSpotrsBatched_bufferSize_(handle,uplo,n,nrhs,A,lda,B,ldb,lwork,batch_count) bind(c, name="cusolverSpotrsBatched_bufferSize")
#else
    function hipsolverSpotrsBatched_bufferSize_(handle,uplo,n,nrhs,A,lda,B,ldb,lwork,batch_count) bind(c, name="hipsolverSpotrsBatched_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrsBatched_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr) :: A
      integer(c_int),value :: lda
      type(c_ptr) :: B
      integer(c_int),value :: ldb
      integer(c_int) :: lwork
      integer(c_int),value :: batch_count
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSpotrsBatched_bufferSize_full_rank,&
      hipsolverSpotrsBatched_bufferSize_rank_0,&
      hipsolverSpotrsBatched_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverDpotrsBatched_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverDpotrsBatched_bufferSize_(handle,uplo,n,nrhs,A,lda,B,ldb,lwork,batch_count) bind(c, name="cusolverDpotrsBatched_bufferSize")
#else
    function hipsolverDpotrsBatched_bufferSize_(handle,uplo,n,nrhs,A,lda,B,ldb,lwork,batch_count) bind(c, name="hipsolverDpotrsBatched_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrsBatched_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr) :: A
      integer(c_int),value :: lda
      type(c_ptr) :: B
      integer(c_int),value :: ldb
      integer(c_int) :: lwork
      integer(c_int),value :: batch_count
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDpotrsBatched_bufferSize_full_rank,&
      hipsolverDpotrsBatched_bufferSize_rank_0,&
      hipsolverDpotrsBatched_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverCpotrsBatched_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverCpotrsBatched_bufferSize_(handle,uplo,n,nrhs,A,lda,B,ldb,lwork,batch_count) bind(c, name="cusolverCpotrsBatched_bufferSize")
#else
    function hipsolverCpotrsBatched_bufferSize_(handle,uplo,n,nrhs,A,lda,B,ldb,lwork,batch_count) bind(c, name="hipsolverCpotrsBatched_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrsBatched_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr) :: A
      integer(c_int),value :: lda
      type(c_ptr) :: B
      integer(c_int),value :: ldb
      integer(c_int) :: lwork
      integer(c_int),value :: batch_count
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCpotrsBatched_bufferSize_full_rank,&
      hipsolverCpotrsBatched_bufferSize_rank_0,&
      hipsolverCpotrsBatched_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverZpotrsBatched_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverZpotrsBatched_bufferSize_(handle,uplo,n,nrhs,A,lda,B,ldb,lwork,batch_count) bind(c, name="cusolverZpotrsBatched_bufferSize")
#else
    function hipsolverZpotrsBatched_bufferSize_(handle,uplo,n,nrhs,A,lda,B,ldb,lwork,batch_count) bind(c, name="hipsolverZpotrsBatched_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrsBatched_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr) :: A
      integer(c_int),value :: lda
      type(c_ptr) :: B
      integer(c_int),value :: ldb
      integer(c_int) :: lwork
      integer(c_int),value :: batch_count
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZpotrsBatched_bufferSize_full_rank,&
      hipsolverZpotrsBatched_bufferSize_rank_0,&
      hipsolverZpotrsBatched_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverSpotrsBatched
#ifdef USE_CUDA_NAMES
    function hipsolverSpotrsBatched_(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo,batch_count) bind(c, name="cusolverSpotrsBatched")
#else
    function hipsolverSpotrsBatched_(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo,batch_count) bind(c, name="hipsolverSpotrsBatched")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrsBatched_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr) :: A
      integer(c_int),value :: lda
      type(c_ptr) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
      integer(c_int),value :: batch_count
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSpotrsBatched_full_rank,&
      hipsolverSpotrsBatched_rank_0,&
      hipsolverSpotrsBatched_rank_1
#endif
  end interface
  
  interface hipsolverDpotrsBatched
#ifdef USE_CUDA_NAMES
    function hipsolverDpotrsBatched_(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo,batch_count) bind(c, name="cusolverDpotrsBatched")
#else
    function hipsolverDpotrsBatched_(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo,batch_count) bind(c, name="hipsolverDpotrsBatched")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrsBatched_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr) :: A
      integer(c_int),value :: lda
      type(c_ptr) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
      integer(c_int),value :: batch_count
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDpotrsBatched_full_rank,&
      hipsolverDpotrsBatched_rank_0,&
      hipsolverDpotrsBatched_rank_1
#endif
  end interface
  
  interface hipsolverCpotrsBatched
#ifdef USE_CUDA_NAMES
    function hipsolverCpotrsBatched_(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo,batch_count) bind(c, name="cusolverCpotrsBatched")
#else
    function hipsolverCpotrsBatched_(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo,batch_count) bind(c, name="hipsolverCpotrsBatched")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrsBatched_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr) :: A
      integer(c_int),value :: lda
      type(c_ptr) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
      integer(c_int),value :: batch_count
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCpotrsBatched_full_rank,&
      hipsolverCpotrsBatched_rank_0,&
      hipsolverCpotrsBatched_rank_1
#endif
  end interface
  
  interface hipsolverZpotrsBatched
#ifdef USE_CUDA_NAMES
    function hipsolverZpotrsBatched_(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo,batch_count) bind(c, name="cusolverZpotrsBatched")
#else
    function hipsolverZpotrsBatched_(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo,batch_count) bind(c, name="hipsolverZpotrsBatched")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrsBatched_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      integer(c_int),value :: nrhs
      type(c_ptr) :: A
      integer(c_int),value :: lda
      type(c_ptr) :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
      integer(c_int),value :: batch_count
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZpotrsBatched_full_rank,&
      hipsolverZpotrsBatched_rank_0,&
      hipsolverZpotrsBatched_rank_1
#endif
  end interface
  
  interface hipsolverSsyevd_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverSsyevd_bufferSize_(handle,jobz,uplo,n,A,lda,D,lwork) bind(c, name="cusolverSsyevd_bufferSize")
#else
    function hipsolverSsyevd_bufferSize_(handle,jobz,uplo,n,A,lda,D,lwork) bind(c, name="hipsolverSsyevd_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsyevd_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)),value :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: D
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSsyevd_bufferSize_full_rank,&
      hipsolverSsyevd_bufferSize_rank_0,&
      hipsolverSsyevd_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverDsyevd_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverDsyevd_bufferSize_(handle,jobz,uplo,n,A,lda,D,lwork) bind(c, name="cusolverDsyevd_bufferSize")
#else
    function hipsolverDsyevd_bufferSize_(handle,jobz,uplo,n,A,lda,D,lwork) bind(c, name="hipsolverDsyevd_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsyevd_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)),value :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: D
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDsyevd_bufferSize_full_rank,&
      hipsolverDsyevd_bufferSize_rank_0,&
      hipsolverDsyevd_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverCheevd_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverCheevd_bufferSize_(handle,jobz,uplo,n,A,lda,D,lwork) bind(c, name="cusolverCheevd_bufferSize")
#else
    function hipsolverCheevd_bufferSize_(handle,jobz,uplo,n,A,lda,D,lwork) bind(c, name="hipsolverCheevd_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCheevd_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)),value :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: D
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCheevd_bufferSize_full_rank,&
      hipsolverCheevd_bufferSize_rank_0,&
      hipsolverCheevd_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverZheevd_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverZheevd_bufferSize_(handle,jobz,uplo,n,A,lda,D,lwork) bind(c, name="cusolverZheevd_bufferSize")
#else
    function hipsolverZheevd_bufferSize_(handle,jobz,uplo,n,A,lda,D,lwork) bind(c, name="hipsolverZheevd_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZheevd_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)),value :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: D
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZheevd_bufferSize_full_rank,&
      hipsolverZheevd_bufferSize_rank_0,&
      hipsolverZheevd_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverSsyevd
#ifdef USE_CUDA_NAMES
    function hipsolverSsyevd_(handle,jobz,uplo,n,A,lda,D,work,lwork,devInfo) bind(c, name="cusolverSsyevd")
#else
    function hipsolverSsyevd_(handle,jobz,uplo,n,A,lda,D,work,lwork,devInfo) bind(c, name="hipsolverSsyevd")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsyevd_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)),value :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: D
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSsyevd_full_rank,&
      hipsolverSsyevd_rank_0,&
      hipsolverSsyevd_rank_1
#endif
  end interface
  
  interface hipsolverDsyevd
#ifdef USE_CUDA_NAMES
    function hipsolverDsyevd_(handle,jobz,uplo,n,A,lda,D,work,lwork,devInfo) bind(c, name="cusolverDsyevd")
#else
    function hipsolverDsyevd_(handle,jobz,uplo,n,A,lda,D,work,lwork,devInfo) bind(c, name="hipsolverDsyevd")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsyevd_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)),value :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: D
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDsyevd_full_rank,&
      hipsolverDsyevd_rank_0,&
      hipsolverDsyevd_rank_1
#endif
  end interface
  
  interface hipsolverCheevd
#ifdef USE_CUDA_NAMES
    function hipsolverCheevd_(handle,jobz,uplo,n,A,lda,D,work,lwork,devInfo) bind(c, name="cusolverCheevd")
#else
    function hipsolverCheevd_(handle,jobz,uplo,n,A,lda,D,work,lwork,devInfo) bind(c, name="hipsolverCheevd")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCheevd_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)),value :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: D
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCheevd_full_rank,&
      hipsolverCheevd_rank_0,&
      hipsolverCheevd_rank_1
#endif
  end interface
  
  interface hipsolverZheevd
#ifdef USE_CUDA_NAMES
    function hipsolverZheevd_(handle,jobz,uplo,n,A,lda,D,work,lwork,devInfo) bind(c, name="cusolverZheevd")
#else
    function hipsolverZheevd_(handle,jobz,uplo,n,A,lda,D,work,lwork,devInfo) bind(c, name="hipsolverZheevd")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZheevd_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)),value :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: D
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZheevd_full_rank,&
      hipsolverZheevd_rank_0,&
      hipsolverZheevd_rank_1
#endif
  end interface
  
  interface hipsolverSsygvd_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverSsygvd_bufferSize_(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,lwork) bind(c, name="cusolverSsygvd_bufferSize")
#else
    function hipsolverSsygvd_bufferSize_(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,lwork) bind(c, name="hipsolverSsygvd_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsygvd_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)),value :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)),value :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: D
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSsygvd_bufferSize_full_rank,&
      hipsolverSsygvd_bufferSize_rank_0,&
      hipsolverSsygvd_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverDsygvd_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverDsygvd_bufferSize_(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,lwork) bind(c, name="cusolverDsygvd_bufferSize")
#else
    function hipsolverDsygvd_bufferSize_(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,lwork) bind(c, name="hipsolverDsygvd_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsygvd_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)),value :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)),value :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: D
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDsygvd_bufferSize_full_rank,&
      hipsolverDsygvd_bufferSize_rank_0,&
      hipsolverDsygvd_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverChegvd_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverChegvd_bufferSize_(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,lwork) bind(c, name="cusolverChegvd_bufferSize")
#else
    function hipsolverChegvd_bufferSize_(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,lwork) bind(c, name="hipsolverChegvd_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverChegvd_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)),value :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)),value :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: D
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverChegvd_bufferSize_full_rank,&
      hipsolverChegvd_bufferSize_rank_0,&
      hipsolverChegvd_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverZhegvd_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverZhegvd_bufferSize_(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,lwork) bind(c, name="cusolverZhegvd_bufferSize")
#else
    function hipsolverZhegvd_bufferSize_(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,lwork) bind(c, name="hipsolverZhegvd_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZhegvd_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)),value :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)),value :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: D
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZhegvd_bufferSize_full_rank,&
      hipsolverZhegvd_bufferSize_rank_0,&
      hipsolverZhegvd_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverSsygvd
#ifdef USE_CUDA_NAMES
    function hipsolverSsygvd_(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,work,lwork,devInfo) bind(c, name="cusolverSsygvd")
#else
    function hipsolverSsygvd_(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,work,lwork,devInfo) bind(c, name="hipsolverSsygvd")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsygvd_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)),value :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)),value :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: D
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSsygvd_full_rank,&
      hipsolverSsygvd_rank_0,&
      hipsolverSsygvd_rank_1
#endif
  end interface
  
  interface hipsolverDsygvd
#ifdef USE_CUDA_NAMES
    function hipsolverDsygvd_(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,work,lwork,devInfo) bind(c, name="cusolverDsygvd")
#else
    function hipsolverDsygvd_(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,work,lwork,devInfo) bind(c, name="hipsolverDsygvd")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsygvd_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)),value :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)),value :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: D
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDsygvd_full_rank,&
      hipsolverDsygvd_rank_0,&
      hipsolverDsygvd_rank_1
#endif
  end interface
  
  interface hipsolverChegvd
#ifdef USE_CUDA_NAMES
    function hipsolverChegvd_(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,work,lwork,devInfo) bind(c, name="cusolverChegvd")
#else
    function hipsolverChegvd_(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,work,lwork,devInfo) bind(c, name="hipsolverChegvd")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverChegvd_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)),value :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)),value :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: D
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverChegvd_full_rank,&
      hipsolverChegvd_rank_0,&
      hipsolverChegvd_rank_1
#endif
  end interface
  
  interface hipsolverZhegvd
#ifdef USE_CUDA_NAMES
    function hipsolverZhegvd_(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,work,lwork,devInfo) bind(c, name="cusolverZhegvd")
#else
    function hipsolverZhegvd_(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,work,lwork,devInfo) bind(c, name="hipsolverZhegvd")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZhegvd_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)),value :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)),value :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: B
      integer(c_int),value :: ldb
      type(c_ptr),value :: D
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZhegvd_full_rank,&
      hipsolverZhegvd_rank_0,&
      hipsolverZhegvd_rank_1
#endif
  end interface
  
  interface hipsolverSsytrd_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverSsytrd_bufferSize_(handle,uplo,n,A,lda,D,E,tau,lwork) bind(c, name="cusolverSsytrd_bufferSize")
#else
    function hipsolverSsytrd_bufferSize_(handle,uplo,n,A,lda,D,E,tau,lwork) bind(c, name="hipsolverSsytrd_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsytrd_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: D
      type(c_ptr),value :: E
      real(c_float) :: tau
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSsytrd_bufferSize_full_rank,&
      hipsolverSsytrd_bufferSize_rank_0,&
      hipsolverSsytrd_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverDsytrd_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverDsytrd_bufferSize_(handle,uplo,n,A,lda,D,E,tau,lwork) bind(c, name="cusolverDsytrd_bufferSize")
#else
    function hipsolverDsytrd_bufferSize_(handle,uplo,n,A,lda,D,E,tau,lwork) bind(c, name="hipsolverDsytrd_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsytrd_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: D
      type(c_ptr),value :: E
      real(c_double) :: tau
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDsytrd_bufferSize_full_rank,&
      hipsolverDsytrd_bufferSize_rank_0,&
      hipsolverDsytrd_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverChetrd_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverChetrd_bufferSize_(handle,uplo,n,A,lda,D,E,tau,lwork) bind(c, name="cusolverChetrd_bufferSize")
#else
    function hipsolverChetrd_bufferSize_(handle,uplo,n,A,lda,D,E,tau,lwork) bind(c, name="hipsolverChetrd_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverChetrd_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: D
      type(c_ptr),value :: E
      complex(c_float_complex) :: tau
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverChetrd_bufferSize_full_rank,&
      hipsolverChetrd_bufferSize_rank_0,&
      hipsolverChetrd_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverZhetrd_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverZhetrd_bufferSize_(handle,uplo,n,A,lda,D,E,tau,lwork) bind(c, name="cusolverZhetrd_bufferSize")
#else
    function hipsolverZhetrd_bufferSize_(handle,uplo,n,A,lda,D,E,tau,lwork) bind(c, name="hipsolverZhetrd_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZhetrd_bufferSize_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: D
      type(c_ptr),value :: E
      complex(c_double_complex) :: tau
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZhetrd_bufferSize_full_rank,&
      hipsolverZhetrd_bufferSize_rank_0,&
      hipsolverZhetrd_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverSsytrd
#ifdef USE_CUDA_NAMES
    function hipsolverSsytrd_(handle,uplo,n,A,lda,D,E,tau,work,lwork,devInfo) bind(c, name="cusolverSsytrd")
#else
    function hipsolverSsytrd_(handle,uplo,n,A,lda,D,E,tau,work,lwork,devInfo) bind(c, name="hipsolverSsytrd")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsytrd_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: D
      type(c_ptr),value :: E
      real(c_float) :: tau
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSsytrd_full_rank,&
      hipsolverSsytrd_rank_0,&
      hipsolverSsytrd_rank_1
#endif
  end interface
  
  interface hipsolverDsytrd
#ifdef USE_CUDA_NAMES
    function hipsolverDsytrd_(handle,uplo,n,A,lda,D,E,tau,work,lwork,devInfo) bind(c, name="cusolverDsytrd")
#else
    function hipsolverDsytrd_(handle,uplo,n,A,lda,D,E,tau,work,lwork,devInfo) bind(c, name="hipsolverDsytrd")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsytrd_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: D
      type(c_ptr),value :: E
      real(c_double) :: tau
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDsytrd_full_rank,&
      hipsolverDsytrd_rank_0,&
      hipsolverDsytrd_rank_1
#endif
  end interface
  
  interface hipsolverChetrd
#ifdef USE_CUDA_NAMES
    function hipsolverChetrd_(handle,uplo,n,A,lda,D,E,tau,work,lwork,devInfo) bind(c, name="cusolverChetrd")
#else
    function hipsolverChetrd_(handle,uplo,n,A,lda,D,E,tau,work,lwork,devInfo) bind(c, name="hipsolverChetrd")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverChetrd_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: D
      type(c_ptr),value :: E
      complex(c_float_complex) :: tau
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverChetrd_full_rank,&
      hipsolverChetrd_rank_0,&
      hipsolverChetrd_rank_1
#endif
  end interface
  
  interface hipsolverZhetrd
#ifdef USE_CUDA_NAMES
    function hipsolverZhetrd_(handle,uplo,n,A,lda,D,E,tau,work,lwork,devInfo) bind(c, name="cusolverZhetrd")
#else
    function hipsolverZhetrd_(handle,uplo,n,A,lda,D,E,tau,work,lwork,devInfo) bind(c, name="hipsolverZhetrd")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZhetrd_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: D
      type(c_ptr),value :: E
      complex(c_double_complex) :: tau
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZhetrd_full_rank,&
      hipsolverZhetrd_rank_0,&
      hipsolverZhetrd_rank_1
#endif
  end interface
  
  interface hipsolverSsytrf_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverSsytrf_bufferSize_(handle,n,A,lda,lwork) bind(c, name="cusolverSsytrf_bufferSize")
#else
    function hipsolverSsytrf_bufferSize_(handle,n,A,lda,lwork) bind(c, name="hipsolverSsytrf_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsytrf_bufferSize_
      type(c_ptr),value :: handle
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSsytrf_bufferSize_full_rank,&
      hipsolverSsytrf_bufferSize_rank_0,&
      hipsolverSsytrf_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverDsytrf_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverDsytrf_bufferSize_(handle,n,A,lda,lwork) bind(c, name="cusolverDsytrf_bufferSize")
#else
    function hipsolverDsytrf_bufferSize_(handle,n,A,lda,lwork) bind(c, name="hipsolverDsytrf_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsytrf_bufferSize_
      type(c_ptr),value :: handle
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDsytrf_bufferSize_full_rank,&
      hipsolverDsytrf_bufferSize_rank_0,&
      hipsolverDsytrf_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverCsytrf_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverCsytrf_bufferSize_(handle,n,A,lda,lwork) bind(c, name="cusolverCsytrf_bufferSize")
#else
    function hipsolverCsytrf_bufferSize_(handle,n,A,lda,lwork) bind(c, name="hipsolverCsytrf_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCsytrf_bufferSize_
      type(c_ptr),value :: handle
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCsytrf_bufferSize_full_rank,&
      hipsolverCsytrf_bufferSize_rank_0,&
      hipsolverCsytrf_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverZsytrf_bufferSize
#ifdef USE_CUDA_NAMES
    function hipsolverZsytrf_bufferSize_(handle,n,A,lda,lwork) bind(c, name="cusolverZsytrf_bufferSize")
#else
    function hipsolverZsytrf_bufferSize_(handle,n,A,lda,lwork) bind(c, name="hipsolverZsytrf_bufferSize")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZsytrf_bufferSize_
      type(c_ptr),value :: handle
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      integer(c_int) :: lwork
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZsytrf_bufferSize_full_rank,&
      hipsolverZsytrf_bufferSize_rank_0,&
      hipsolverZsytrf_bufferSize_rank_1
#endif
  end interface
  
  interface hipsolverSsytrf
#ifdef USE_CUDA_NAMES
    function hipsolverSsytrf_(handle,uplo,n,A,lda,ipiv,work,lwork,devInfo) bind(c, name="cusolverSsytrf")
#else
    function hipsolverSsytrf_(handle,uplo,n,A,lda,ipiv,work,lwork,devInfo) bind(c, name="hipsolverSsytrf")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsytrf_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: ipiv
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverSsytrf_full_rank,&
      hipsolverSsytrf_rank_0,&
      hipsolverSsytrf_rank_1
#endif
  end interface
  
  interface hipsolverDsytrf
#ifdef USE_CUDA_NAMES
    function hipsolverDsytrf_(handle,uplo,n,A,lda,ipiv,work,lwork,devInfo) bind(c, name="cusolverDsytrf")
#else
    function hipsolverDsytrf_(handle,uplo,n,A,lda,ipiv,work,lwork,devInfo) bind(c, name="hipsolverDsytrf")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsytrf_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: ipiv
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverDsytrf_full_rank,&
      hipsolverDsytrf_rank_0,&
      hipsolverDsytrf_rank_1
#endif
  end interface
  
  interface hipsolverCsytrf
#ifdef USE_CUDA_NAMES
    function hipsolverCsytrf_(handle,uplo,n,A,lda,ipiv,work,lwork,devInfo) bind(c, name="cusolverCsytrf")
#else
    function hipsolverCsytrf_(handle,uplo,n,A,lda,ipiv,work,lwork,devInfo) bind(c, name="hipsolverCsytrf")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCsytrf_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: ipiv
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverCsytrf_full_rank,&
      hipsolverCsytrf_rank_0,&
      hipsolverCsytrf_rank_1
#endif
  end interface
  
  interface hipsolverZsytrf
#ifdef USE_CUDA_NAMES
    function hipsolverZsytrf_(handle,uplo,n,A,lda,ipiv,work,lwork,devInfo) bind(c, name="cusolverZsytrf")
#else
    function hipsolverZsytrf_(handle,uplo,n,A,lda,ipiv,work,lwork,devInfo) bind(c, name="hipsolverZsytrf")
#endif
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZsytrf_
      type(c_ptr),value :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)),value :: uplo
      integer(c_int),value :: n
      type(c_ptr),value :: A
      integer(c_int),value :: lda
      type(c_ptr),value :: ipiv
      type(c_ptr),value :: work
      integer(c_int),value :: lwork
      integer(c_int) :: devInfo
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipsolverZsytrf_full_rank,&
      hipsolverZsytrf_rank_0,&
      hipsolverZsytrf_rank_1
#endif
  end interface

#ifdef USE_FPOINTER_INTERFACES
  contains
    function hipsolverSorgbr_bufferSize_full_rank(handle,side,m,n,k,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgbr_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      integer(c_int) :: lwork
      !
      hipsolverSorgbr_bufferSize_full_rank = hipsolverSorgbr_bufferSize_(handle,side,m,n,k,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverSorgbr_bufferSize_rank_0(handle,side,m,n,k,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgbr_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_float),target :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      integer(c_int) :: lwork
      !
      hipsolverSorgbr_bufferSize_rank_0 = hipsolverSorgbr_bufferSize_(handle,side,m,n,k,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverSorgbr_bufferSize_rank_1(handle,side,m,n,k,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgbr_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      integer(c_int) :: lwork
      !
      hipsolverSorgbr_bufferSize_rank_1 = hipsolverSorgbr_bufferSize_(handle,side,m,n,k,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverDorgbr_bufferSize_full_rank(handle,side,m,n,k,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgbr_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      integer(c_int) :: lwork
      !
      hipsolverDorgbr_bufferSize_full_rank = hipsolverDorgbr_bufferSize_(handle,side,m,n,k,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverDorgbr_bufferSize_rank_0(handle,side,m,n,k,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgbr_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_double),target :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      integer(c_int) :: lwork
      !
      hipsolverDorgbr_bufferSize_rank_0 = hipsolverDorgbr_bufferSize_(handle,side,m,n,k,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverDorgbr_bufferSize_rank_1(handle,side,m,n,k,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgbr_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      integer(c_int) :: lwork
      !
      hipsolverDorgbr_bufferSize_rank_1 = hipsolverDorgbr_bufferSize_(handle,side,m,n,k,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverCungbr_bufferSize_full_rank(handle,side,m,n,k,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungbr_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      integer(c_int) :: lwork
      !
      hipsolverCungbr_bufferSize_full_rank = hipsolverCungbr_bufferSize_(handle,side,m,n,k,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverCungbr_bufferSize_rank_0(handle,side,m,n,k,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungbr_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      integer(c_int) :: lwork
      !
      hipsolverCungbr_bufferSize_rank_0 = hipsolverCungbr_bufferSize_(handle,side,m,n,k,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverCungbr_bufferSize_rank_1(handle,side,m,n,k,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungbr_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      integer(c_int) :: lwork
      !
      hipsolverCungbr_bufferSize_rank_1 = hipsolverCungbr_bufferSize_(handle,side,m,n,k,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverZungbr_bufferSize_full_rank(handle,side,m,n,k,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungbr_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      integer(c_int) :: lwork
      !
      hipsolverZungbr_bufferSize_full_rank = hipsolverZungbr_bufferSize_(handle,side,m,n,k,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverZungbr_bufferSize_rank_0(handle,side,m,n,k,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungbr_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      integer(c_int) :: lwork
      !
      hipsolverZungbr_bufferSize_rank_0 = hipsolverZungbr_bufferSize_(handle,side,m,n,k,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverZungbr_bufferSize_rank_1(handle,side,m,n,k,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungbr_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      integer(c_int) :: lwork
      !
      hipsolverZungbr_bufferSize_rank_1 = hipsolverZungbr_bufferSize_(handle,side,m,n,k,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverSorgbr_full_rank(handle,side,m,n,k,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgbr_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSorgbr_full_rank = hipsolverSorgbr_(handle,side,m,n,k,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverSorgbr_rank_0(handle,side,m,n,k,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgbr_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_float),target :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSorgbr_rank_0 = hipsolverSorgbr_(handle,side,m,n,k,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverSorgbr_rank_1(handle,side,m,n,k,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgbr_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSorgbr_rank_1 = hipsolverSorgbr_(handle,side,m,n,k,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverDorgbr_full_rank(handle,side,m,n,k,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgbr_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDorgbr_full_rank = hipsolverDorgbr_(handle,side,m,n,k,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverDorgbr_rank_0(handle,side,m,n,k,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgbr_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_double),target :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDorgbr_rank_0 = hipsolverDorgbr_(handle,side,m,n,k,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverDorgbr_rank_1(handle,side,m,n,k,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgbr_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDorgbr_rank_1 = hipsolverDorgbr_(handle,side,m,n,k,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverCungbr_full_rank(handle,side,m,n,k,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungbr_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCungbr_full_rank = hipsolverCungbr_(handle,side,m,n,k,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverCungbr_rank_0(handle,side,m,n,k,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungbr_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCungbr_rank_0 = hipsolverCungbr_(handle,side,m,n,k,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverCungbr_rank_1(handle,side,m,n,k,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungbr_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCungbr_rank_1 = hipsolverCungbr_(handle,side,m,n,k,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverZungbr_full_rank(handle,side,m,n,k,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungbr_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZungbr_full_rank = hipsolverZungbr_(handle,side,m,n,k,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverZungbr_rank_0(handle,side,m,n,k,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungbr_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZungbr_rank_0 = hipsolverZungbr_(handle,side,m,n,k,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverZungbr_rank_1(handle,side,m,n,k,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungbr_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZungbr_rank_1 = hipsolverZungbr_(handle,side,m,n,k,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverSorgqr_bufferSize_full_rank(handle,m,n,k,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgqr_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      integer(c_int) :: lwork
      !
      hipsolverSorgqr_bufferSize_full_rank = hipsolverSorgqr_bufferSize_(handle,m,n,k,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverSorgqr_bufferSize_rank_0(handle,m,n,k,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgqr_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_float),target :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      integer(c_int) :: lwork
      !
      hipsolverSorgqr_bufferSize_rank_0 = hipsolverSorgqr_bufferSize_(handle,m,n,k,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverSorgqr_bufferSize_rank_1(handle,m,n,k,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgqr_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      integer(c_int) :: lwork
      !
      hipsolverSorgqr_bufferSize_rank_1 = hipsolverSorgqr_bufferSize_(handle,m,n,k,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverDorgqr_bufferSize_full_rank(handle,m,n,k,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgqr_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      integer(c_int) :: lwork
      !
      hipsolverDorgqr_bufferSize_full_rank = hipsolverDorgqr_bufferSize_(handle,m,n,k,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverDorgqr_bufferSize_rank_0(handle,m,n,k,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgqr_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_double),target :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      integer(c_int) :: lwork
      !
      hipsolverDorgqr_bufferSize_rank_0 = hipsolverDorgqr_bufferSize_(handle,m,n,k,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverDorgqr_bufferSize_rank_1(handle,m,n,k,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgqr_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      integer(c_int) :: lwork
      !
      hipsolverDorgqr_bufferSize_rank_1 = hipsolverDorgqr_bufferSize_(handle,m,n,k,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverCungqr_bufferSize_full_rank(handle,m,n,k,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungqr_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      integer(c_int) :: lwork
      !
      hipsolverCungqr_bufferSize_full_rank = hipsolverCungqr_bufferSize_(handle,m,n,k,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverCungqr_bufferSize_rank_0(handle,m,n,k,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungqr_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      integer(c_int) :: lwork
      !
      hipsolverCungqr_bufferSize_rank_0 = hipsolverCungqr_bufferSize_(handle,m,n,k,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverCungqr_bufferSize_rank_1(handle,m,n,k,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungqr_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      integer(c_int) :: lwork
      !
      hipsolverCungqr_bufferSize_rank_1 = hipsolverCungqr_bufferSize_(handle,m,n,k,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverZungqr_bufferSize_full_rank(handle,m,n,k,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungqr_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      integer(c_int) :: lwork
      !
      hipsolverZungqr_bufferSize_full_rank = hipsolverZungqr_bufferSize_(handle,m,n,k,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverZungqr_bufferSize_rank_0(handle,m,n,k,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungqr_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      integer(c_int) :: lwork
      !
      hipsolverZungqr_bufferSize_rank_0 = hipsolverZungqr_bufferSize_(handle,m,n,k,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverZungqr_bufferSize_rank_1(handle,m,n,k,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungqr_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      integer(c_int) :: lwork
      !
      hipsolverZungqr_bufferSize_rank_1 = hipsolverZungqr_bufferSize_(handle,m,n,k,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverSorgqr_full_rank(handle,m,n,k,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgqr_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSorgqr_full_rank = hipsolverSorgqr_(handle,m,n,k,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverSorgqr_rank_0(handle,m,n,k,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgqr_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_float),target :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSorgqr_rank_0 = hipsolverSorgqr_(handle,m,n,k,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverSorgqr_rank_1(handle,m,n,k,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgqr_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSorgqr_rank_1 = hipsolverSorgqr_(handle,m,n,k,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverDorgqr_full_rank(handle,m,n,k,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgqr_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDorgqr_full_rank = hipsolverDorgqr_(handle,m,n,k,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverDorgqr_rank_0(handle,m,n,k,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgqr_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_double),target :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDorgqr_rank_0 = hipsolverDorgqr_(handle,m,n,k,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverDorgqr_rank_1(handle,m,n,k,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgqr_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDorgqr_rank_1 = hipsolverDorgqr_(handle,m,n,k,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverCungqr_full_rank(handle,m,n,k,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungqr_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCungqr_full_rank = hipsolverCungqr_(handle,m,n,k,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverCungqr_rank_0(handle,m,n,k,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungqr_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCungqr_rank_0 = hipsolverCungqr_(handle,m,n,k,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverCungqr_rank_1(handle,m,n,k,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungqr_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCungqr_rank_1 = hipsolverCungqr_(handle,m,n,k,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverZungqr_full_rank(handle,m,n,k,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungqr_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZungqr_full_rank = hipsolverZungqr_(handle,m,n,k,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverZungqr_rank_0(handle,m,n,k,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungqr_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZungqr_rank_0 = hipsolverZungqr_(handle,m,n,k,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverZungqr_rank_1(handle,m,n,k,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungqr_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZungqr_rank_1 = hipsolverZungqr_(handle,m,n,k,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverSorgtr_bufferSize_full_rank(handle,uplo,n,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgtr_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      integer(c_int) :: lwork
      !
      hipsolverSorgtr_bufferSize_full_rank = hipsolverSorgtr_bufferSize_(handle,uplo,n,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverSorgtr_bufferSize_rank_0(handle,uplo,n,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgtr_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      integer(c_int) :: lwork
      !
      hipsolverSorgtr_bufferSize_rank_0 = hipsolverSorgtr_bufferSize_(handle,uplo,n,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverSorgtr_bufferSize_rank_1(handle,uplo,n,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgtr_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      integer(c_int) :: lwork
      !
      hipsolverSorgtr_bufferSize_rank_1 = hipsolverSorgtr_bufferSize_(handle,uplo,n,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverDorgtr_bufferSize_full_rank(handle,uplo,n,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgtr_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      integer(c_int) :: lwork
      !
      hipsolverDorgtr_bufferSize_full_rank = hipsolverDorgtr_bufferSize_(handle,uplo,n,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverDorgtr_bufferSize_rank_0(handle,uplo,n,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgtr_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      integer(c_int) :: lwork
      !
      hipsolverDorgtr_bufferSize_rank_0 = hipsolverDorgtr_bufferSize_(handle,uplo,n,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverDorgtr_bufferSize_rank_1(handle,uplo,n,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgtr_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      integer(c_int) :: lwork
      !
      hipsolverDorgtr_bufferSize_rank_1 = hipsolverDorgtr_bufferSize_(handle,uplo,n,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverCungtr_bufferSize_full_rank(handle,uplo,n,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungtr_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      integer(c_int) :: lwork
      !
      hipsolverCungtr_bufferSize_full_rank = hipsolverCungtr_bufferSize_(handle,uplo,n,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverCungtr_bufferSize_rank_0(handle,uplo,n,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungtr_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      integer(c_int) :: lwork
      !
      hipsolverCungtr_bufferSize_rank_0 = hipsolverCungtr_bufferSize_(handle,uplo,n,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverCungtr_bufferSize_rank_1(handle,uplo,n,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungtr_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      integer(c_int) :: lwork
      !
      hipsolverCungtr_bufferSize_rank_1 = hipsolverCungtr_bufferSize_(handle,uplo,n,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverZungtr_bufferSize_full_rank(handle,uplo,n,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungtr_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      integer(c_int) :: lwork
      !
      hipsolverZungtr_bufferSize_full_rank = hipsolverZungtr_bufferSize_(handle,uplo,n,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverZungtr_bufferSize_rank_0(handle,uplo,n,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungtr_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      integer(c_int) :: lwork
      !
      hipsolverZungtr_bufferSize_rank_0 = hipsolverZungtr_bufferSize_(handle,uplo,n,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverZungtr_bufferSize_rank_1(handle,uplo,n,A,lda,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungtr_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      integer(c_int) :: lwork
      !
      hipsolverZungtr_bufferSize_rank_1 = hipsolverZungtr_bufferSize_(handle,uplo,n,c_loc(A),lda,tau,lwork)
    end function

    function hipsolverSorgtr_full_rank(handle,uplo,n,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgtr_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSorgtr_full_rank = hipsolverSorgtr_(handle,uplo,n,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverSorgtr_rank_0(handle,uplo,n,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgtr_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSorgtr_rank_0 = hipsolverSorgtr_(handle,uplo,n,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverSorgtr_rank_1(handle,uplo,n,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgtr_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSorgtr_rank_1 = hipsolverSorgtr_(handle,uplo,n,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverDorgtr_full_rank(handle,uplo,n,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgtr_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDorgtr_full_rank = hipsolverDorgtr_(handle,uplo,n,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverDorgtr_rank_0(handle,uplo,n,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgtr_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDorgtr_rank_0 = hipsolverDorgtr_(handle,uplo,n,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverDorgtr_rank_1(handle,uplo,n,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgtr_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDorgtr_rank_1 = hipsolverDorgtr_(handle,uplo,n,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverCungtr_full_rank(handle,uplo,n,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungtr_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCungtr_full_rank = hipsolverCungtr_(handle,uplo,n,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverCungtr_rank_0(handle,uplo,n,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungtr_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCungtr_rank_0 = hipsolverCungtr_(handle,uplo,n,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverCungtr_rank_1(handle,uplo,n,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungtr_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCungtr_rank_1 = hipsolverCungtr_(handle,uplo,n,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverZungtr_full_rank(handle,uplo,n,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungtr_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZungtr_full_rank = hipsolverZungtr_(handle,uplo,n,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverZungtr_rank_0(handle,uplo,n,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungtr_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZungtr_rank_0 = hipsolverZungtr_(handle,uplo,n,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverZungtr_rank_1(handle,uplo,n,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungtr_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZungtr_rank_1 = hipsolverZungtr_(handle,uplo,n,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverSormqr_bufferSize_full_rank(handle,side,trans,m,n,k,A,lda,tau,C,ldc,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSormqr_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      real(c_float),target,dimension(:,:) :: C
      integer(c_int) :: ldc
      integer(c_int) :: lwork
      !
      hipsolverSormqr_bufferSize_full_rank = hipsolverSormqr_bufferSize_(handle,side,trans,m,n,k,c_loc(A),lda,tau,c_loc(C),ldc,lwork)
    end function

    function hipsolverSormqr_bufferSize_rank_0(handle,side,trans,m,n,k,A,lda,tau,C,ldc,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSormqr_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_float),target :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      real(c_float),target :: C
      integer(c_int) :: ldc
      integer(c_int) :: lwork
      !
      hipsolverSormqr_bufferSize_rank_0 = hipsolverSormqr_bufferSize_(handle,side,trans,m,n,k,c_loc(A),lda,tau,c_loc(C),ldc,lwork)
    end function

    function hipsolverSormqr_bufferSize_rank_1(handle,side,trans,m,n,k,A,lda,tau,C,ldc,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSormqr_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      real(c_float),target,dimension(:) :: C
      integer(c_int) :: ldc
      integer(c_int) :: lwork
      !
      hipsolverSormqr_bufferSize_rank_1 = hipsolverSormqr_bufferSize_(handle,side,trans,m,n,k,c_loc(A),lda,tau,c_loc(C),ldc,lwork)
    end function

    function hipsolverDormqr_bufferSize_full_rank(handle,side,trans,m,n,k,A,lda,tau,C,ldc,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDormqr_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      real(c_double),target,dimension(:,:) :: C
      integer(c_int) :: ldc
      integer(c_int) :: lwork
      !
      hipsolverDormqr_bufferSize_full_rank = hipsolverDormqr_bufferSize_(handle,side,trans,m,n,k,c_loc(A),lda,tau,c_loc(C),ldc,lwork)
    end function

    function hipsolverDormqr_bufferSize_rank_0(handle,side,trans,m,n,k,A,lda,tau,C,ldc,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDormqr_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_double),target :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      real(c_double),target :: C
      integer(c_int) :: ldc
      integer(c_int) :: lwork
      !
      hipsolverDormqr_bufferSize_rank_0 = hipsolverDormqr_bufferSize_(handle,side,trans,m,n,k,c_loc(A),lda,tau,c_loc(C),ldc,lwork)
    end function

    function hipsolverDormqr_bufferSize_rank_1(handle,side,trans,m,n,k,A,lda,tau,C,ldc,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDormqr_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      real(c_double),target,dimension(:) :: C
      integer(c_int) :: ldc
      integer(c_int) :: lwork
      !
      hipsolverDormqr_bufferSize_rank_1 = hipsolverDormqr_bufferSize_(handle,side,trans,m,n,k,c_loc(A),lda,tau,c_loc(C),ldc,lwork)
    end function

    function hipsolverCunmqr_bufferSize_full_rank(handle,side,trans,m,n,k,A,lda,tau,C,ldc,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCunmqr_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      complex(c_float_complex),target,dimension(:,:) :: C
      integer(c_int) :: ldc
      integer(c_int) :: lwork
      !
      hipsolverCunmqr_bufferSize_full_rank = hipsolverCunmqr_bufferSize_(handle,side,trans,m,n,k,c_loc(A),lda,tau,c_loc(C),ldc,lwork)
    end function

    function hipsolverCunmqr_bufferSize_rank_0(handle,side,trans,m,n,k,A,lda,tau,C,ldc,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCunmqr_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      complex(c_float_complex),target :: C
      integer(c_int) :: ldc
      integer(c_int) :: lwork
      !
      hipsolverCunmqr_bufferSize_rank_0 = hipsolverCunmqr_bufferSize_(handle,side,trans,m,n,k,c_loc(A),lda,tau,c_loc(C),ldc,lwork)
    end function

    function hipsolverCunmqr_bufferSize_rank_1(handle,side,trans,m,n,k,A,lda,tau,C,ldc,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCunmqr_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      complex(c_float_complex),target,dimension(:) :: C
      integer(c_int) :: ldc
      integer(c_int) :: lwork
      !
      hipsolverCunmqr_bufferSize_rank_1 = hipsolverCunmqr_bufferSize_(handle,side,trans,m,n,k,c_loc(A),lda,tau,c_loc(C),ldc,lwork)
    end function

    function hipsolverZunmqr_bufferSize_full_rank(handle,side,trans,m,n,k,A,lda,tau,C,ldc,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZunmqr_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      complex(c_double_complex),target,dimension(:,:) :: C
      integer(c_int) :: ldc
      integer(c_int) :: lwork
      !
      hipsolverZunmqr_bufferSize_full_rank = hipsolverZunmqr_bufferSize_(handle,side,trans,m,n,k,c_loc(A),lda,tau,c_loc(C),ldc,lwork)
    end function

    function hipsolverZunmqr_bufferSize_rank_0(handle,side,trans,m,n,k,A,lda,tau,C,ldc,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZunmqr_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      complex(c_double_complex),target :: C
      integer(c_int) :: ldc
      integer(c_int) :: lwork
      !
      hipsolverZunmqr_bufferSize_rank_0 = hipsolverZunmqr_bufferSize_(handle,side,trans,m,n,k,c_loc(A),lda,tau,c_loc(C),ldc,lwork)
    end function

    function hipsolverZunmqr_bufferSize_rank_1(handle,side,trans,m,n,k,A,lda,tau,C,ldc,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZunmqr_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      complex(c_double_complex),target,dimension(:) :: C
      integer(c_int) :: ldc
      integer(c_int) :: lwork
      !
      hipsolverZunmqr_bufferSize_rank_1 = hipsolverZunmqr_bufferSize_(handle,side,trans,m,n,k,c_loc(A),lda,tau,c_loc(C),ldc,lwork)
    end function

    function hipsolverSormqr_full_rank(handle,side,trans,m,n,k,A,lda,tau,C,ldc,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSormqr_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      real(c_float),target,dimension(:,:) :: C
      integer(c_int) :: ldc
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSormqr_full_rank = hipsolverSormqr_(handle,side,trans,m,n,k,c_loc(A),lda,tau,c_loc(C),ldc,work,lwork,devInfo)
    end function

    function hipsolverSormqr_rank_0(handle,side,trans,m,n,k,A,lda,tau,C,ldc,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSormqr_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_float),target :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      real(c_float),target :: C
      integer(c_int) :: ldc
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSormqr_rank_0 = hipsolverSormqr_(handle,side,trans,m,n,k,c_loc(A),lda,tau,c_loc(C),ldc,work,lwork,devInfo)
    end function

    function hipsolverSormqr_rank_1(handle,side,trans,m,n,k,A,lda,tau,C,ldc,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSormqr_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      real(c_float),target,dimension(:) :: C
      integer(c_int) :: ldc
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSormqr_rank_1 = hipsolverSormqr_(handle,side,trans,m,n,k,c_loc(A),lda,tau,c_loc(C),ldc,work,lwork,devInfo)
    end function

    function hipsolverDormqr_full_rank(handle,side,trans,m,n,k,A,lda,tau,C,ldc,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDormqr_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      real(c_double),target,dimension(:,:) :: C
      integer(c_int) :: ldc
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDormqr_full_rank = hipsolverDormqr_(handle,side,trans,m,n,k,c_loc(A),lda,tau,c_loc(C),ldc,work,lwork,devInfo)
    end function

    function hipsolverDormqr_rank_0(handle,side,trans,m,n,k,A,lda,tau,C,ldc,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDormqr_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_double),target :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      real(c_double),target :: C
      integer(c_int) :: ldc
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDormqr_rank_0 = hipsolverDormqr_(handle,side,trans,m,n,k,c_loc(A),lda,tau,c_loc(C),ldc,work,lwork,devInfo)
    end function

    function hipsolverDormqr_rank_1(handle,side,trans,m,n,k,A,lda,tau,C,ldc,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDormqr_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      real(c_double),target,dimension(:) :: C
      integer(c_int) :: ldc
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDormqr_rank_1 = hipsolverDormqr_(handle,side,trans,m,n,k,c_loc(A),lda,tau,c_loc(C),ldc,work,lwork,devInfo)
    end function

    function hipsolverCunmqr_full_rank(handle,side,trans,m,n,k,A,lda,tau,C,ldc,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCunmqr_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      complex(c_float_complex),target,dimension(:,:) :: C
      integer(c_int) :: ldc
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCunmqr_full_rank = hipsolverCunmqr_(handle,side,trans,m,n,k,c_loc(A),lda,tau,c_loc(C),ldc,work,lwork,devInfo)
    end function

    function hipsolverCunmqr_rank_0(handle,side,trans,m,n,k,A,lda,tau,C,ldc,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCunmqr_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      complex(c_float_complex),target :: C
      integer(c_int) :: ldc
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCunmqr_rank_0 = hipsolverCunmqr_(handle,side,trans,m,n,k,c_loc(A),lda,tau,c_loc(C),ldc,work,lwork,devInfo)
    end function

    function hipsolverCunmqr_rank_1(handle,side,trans,m,n,k,A,lda,tau,C,ldc,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCunmqr_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      complex(c_float_complex),target,dimension(:) :: C
      integer(c_int) :: ldc
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCunmqr_rank_1 = hipsolverCunmqr_(handle,side,trans,m,n,k,c_loc(A),lda,tau,c_loc(C),ldc,work,lwork,devInfo)
    end function

    function hipsolverZunmqr_full_rank(handle,side,trans,m,n,k,A,lda,tau,C,ldc,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZunmqr_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      complex(c_double_complex),target,dimension(:,:) :: C
      integer(c_int) :: ldc
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZunmqr_full_rank = hipsolverZunmqr_(handle,side,trans,m,n,k,c_loc(A),lda,tau,c_loc(C),ldc,work,lwork,devInfo)
    end function

    function hipsolverZunmqr_rank_0(handle,side,trans,m,n,k,A,lda,tau,C,ldc,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZunmqr_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      complex(c_double_complex),target :: C
      integer(c_int) :: ldc
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZunmqr_rank_0 = hipsolverZunmqr_(handle,side,trans,m,n,k,c_loc(A),lda,tau,c_loc(C),ldc,work,lwork,devInfo)
    end function

    function hipsolverZunmqr_rank_1(handle,side,trans,m,n,k,A,lda,tau,C,ldc,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZunmqr_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      integer(c_int) :: k
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      complex(c_double_complex),target,dimension(:) :: C
      integer(c_int) :: ldc
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZunmqr_rank_1 = hipsolverZunmqr_(handle,side,trans,m,n,k,c_loc(A),lda,tau,c_loc(C),ldc,work,lwork,devInfo)
    end function

    function hipsolverSormtr_bufferSize_full_rank(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSormtr_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      real(c_float),target,dimension(:,:) :: C
      integer(c_int) :: ldc
      integer(c_int) :: lwork
      !
      hipsolverSormtr_bufferSize_full_rank = hipsolverSormtr_bufferSize_(handle,side,uplo,trans,m,n,c_loc(A),lda,tau,c_loc(C),ldc,lwork)
    end function

    function hipsolverSormtr_bufferSize_rank_0(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSormtr_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_float),target :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      real(c_float),target :: C
      integer(c_int) :: ldc
      integer(c_int) :: lwork
      !
      hipsolverSormtr_bufferSize_rank_0 = hipsolverSormtr_bufferSize_(handle,side,uplo,trans,m,n,c_loc(A),lda,tau,c_loc(C),ldc,lwork)
    end function

    function hipsolverSormtr_bufferSize_rank_1(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSormtr_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      real(c_float),target,dimension(:) :: C
      integer(c_int) :: ldc
      integer(c_int) :: lwork
      !
      hipsolverSormtr_bufferSize_rank_1 = hipsolverSormtr_bufferSize_(handle,side,uplo,trans,m,n,c_loc(A),lda,tau,c_loc(C),ldc,lwork)
    end function

    function hipsolverDormtr_bufferSize_full_rank(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDormtr_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      real(c_double),target,dimension(:,:) :: C
      integer(c_int) :: ldc
      integer(c_int) :: lwork
      !
      hipsolverDormtr_bufferSize_full_rank = hipsolverDormtr_bufferSize_(handle,side,uplo,trans,m,n,c_loc(A),lda,tau,c_loc(C),ldc,lwork)
    end function

    function hipsolverDormtr_bufferSize_rank_0(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDormtr_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_double),target :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      real(c_double),target :: C
      integer(c_int) :: ldc
      integer(c_int) :: lwork
      !
      hipsolverDormtr_bufferSize_rank_0 = hipsolverDormtr_bufferSize_(handle,side,uplo,trans,m,n,c_loc(A),lda,tau,c_loc(C),ldc,lwork)
    end function

    function hipsolverDormtr_bufferSize_rank_1(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDormtr_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      real(c_double),target,dimension(:) :: C
      integer(c_int) :: ldc
      integer(c_int) :: lwork
      !
      hipsolverDormtr_bufferSize_rank_1 = hipsolverDormtr_bufferSize_(handle,side,uplo,trans,m,n,c_loc(A),lda,tau,c_loc(C),ldc,lwork)
    end function

    function hipsolverCunmtr_bufferSize_full_rank(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCunmtr_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      complex(c_float_complex),target,dimension(:,:) :: C
      integer(c_int) :: ldc
      integer(c_int) :: lwork
      !
      hipsolverCunmtr_bufferSize_full_rank = hipsolverCunmtr_bufferSize_(handle,side,uplo,trans,m,n,c_loc(A),lda,tau,c_loc(C),ldc,lwork)
    end function

    function hipsolverCunmtr_bufferSize_rank_0(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCunmtr_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      complex(c_float_complex),target :: C
      integer(c_int) :: ldc
      integer(c_int) :: lwork
      !
      hipsolverCunmtr_bufferSize_rank_0 = hipsolverCunmtr_bufferSize_(handle,side,uplo,trans,m,n,c_loc(A),lda,tau,c_loc(C),ldc,lwork)
    end function

    function hipsolverCunmtr_bufferSize_rank_1(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCunmtr_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      complex(c_float_complex),target,dimension(:) :: C
      integer(c_int) :: ldc
      integer(c_int) :: lwork
      !
      hipsolverCunmtr_bufferSize_rank_1 = hipsolverCunmtr_bufferSize_(handle,side,uplo,trans,m,n,c_loc(A),lda,tau,c_loc(C),ldc,lwork)
    end function

    function hipsolverZunmtr_bufferSize_full_rank(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZunmtr_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      complex(c_double_complex),target,dimension(:,:) :: C
      integer(c_int) :: ldc
      integer(c_int) :: lwork
      !
      hipsolverZunmtr_bufferSize_full_rank = hipsolverZunmtr_bufferSize_(handle,side,uplo,trans,m,n,c_loc(A),lda,tau,c_loc(C),ldc,lwork)
    end function

    function hipsolverZunmtr_bufferSize_rank_0(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZunmtr_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      complex(c_double_complex),target :: C
      integer(c_int) :: ldc
      integer(c_int) :: lwork
      !
      hipsolverZunmtr_bufferSize_rank_0 = hipsolverZunmtr_bufferSize_(handle,side,uplo,trans,m,n,c_loc(A),lda,tau,c_loc(C),ldc,lwork)
    end function

    function hipsolverZunmtr_bufferSize_rank_1(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZunmtr_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      complex(c_double_complex),target,dimension(:) :: C
      integer(c_int) :: ldc
      integer(c_int) :: lwork
      !
      hipsolverZunmtr_bufferSize_rank_1 = hipsolverZunmtr_bufferSize_(handle,side,uplo,trans,m,n,c_loc(A),lda,tau,c_loc(C),ldc,lwork)
    end function

    function hipsolverSormtr_full_rank(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSormtr_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      real(c_float),target,dimension(:,:) :: C
      integer(c_int) :: ldc
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSormtr_full_rank = hipsolverSormtr_(handle,side,uplo,trans,m,n,c_loc(A),lda,tau,c_loc(C),ldc,work,lwork,devInfo)
    end function

    function hipsolverSormtr_rank_0(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSormtr_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_float),target :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      real(c_float),target :: C
      integer(c_int) :: ldc
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSormtr_rank_0 = hipsolverSormtr_(handle,side,uplo,trans,m,n,c_loc(A),lda,tau,c_loc(C),ldc,work,lwork,devInfo)
    end function

    function hipsolverSormtr_rank_1(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSormtr_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      real(c_float),target,dimension(:) :: C
      integer(c_int) :: ldc
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSormtr_rank_1 = hipsolverSormtr_(handle,side,uplo,trans,m,n,c_loc(A),lda,tau,c_loc(C),ldc,work,lwork,devInfo)
    end function

    function hipsolverDormtr_full_rank(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDormtr_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      real(c_double),target,dimension(:,:) :: C
      integer(c_int) :: ldc
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDormtr_full_rank = hipsolverDormtr_(handle,side,uplo,trans,m,n,c_loc(A),lda,tau,c_loc(C),ldc,work,lwork,devInfo)
    end function

    function hipsolverDormtr_rank_0(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDormtr_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_double),target :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      real(c_double),target :: C
      integer(c_int) :: ldc
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDormtr_rank_0 = hipsolverDormtr_(handle,side,uplo,trans,m,n,c_loc(A),lda,tau,c_loc(C),ldc,work,lwork,devInfo)
    end function

    function hipsolverDormtr_rank_1(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDormtr_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      real(c_double),target,dimension(:) :: C
      integer(c_int) :: ldc
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDormtr_rank_1 = hipsolverDormtr_(handle,side,uplo,trans,m,n,c_loc(A),lda,tau,c_loc(C),ldc,work,lwork,devInfo)
    end function

    function hipsolverCunmtr_full_rank(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCunmtr_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      complex(c_float_complex),target,dimension(:,:) :: C
      integer(c_int) :: ldc
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCunmtr_full_rank = hipsolverCunmtr_(handle,side,uplo,trans,m,n,c_loc(A),lda,tau,c_loc(C),ldc,work,lwork,devInfo)
    end function

    function hipsolverCunmtr_rank_0(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCunmtr_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      complex(c_float_complex),target :: C
      integer(c_int) :: ldc
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCunmtr_rank_0 = hipsolverCunmtr_(handle,side,uplo,trans,m,n,c_loc(A),lda,tau,c_loc(C),ldc,work,lwork,devInfo)
    end function

    function hipsolverCunmtr_rank_1(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCunmtr_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      complex(c_float_complex),target,dimension(:) :: C
      integer(c_int) :: ldc
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCunmtr_rank_1 = hipsolverCunmtr_(handle,side,uplo,trans,m,n,c_loc(A),lda,tau,c_loc(C),ldc,work,lwork,devInfo)
    end function

    function hipsolverZunmtr_full_rank(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZunmtr_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      complex(c_double_complex),target,dimension(:,:) :: C
      integer(c_int) :: ldc
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZunmtr_full_rank = hipsolverZunmtr_(handle,side,uplo,trans,m,n,c_loc(A),lda,tau,c_loc(C),ldc,work,lwork,devInfo)
    end function

    function hipsolverZunmtr_rank_0(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZunmtr_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      complex(c_double_complex),target :: C
      integer(c_int) :: ldc
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZunmtr_rank_0 = hipsolverZunmtr_(handle,side,uplo,trans,m,n,c_loc(A),lda,tau,c_loc(C),ldc,work,lwork,devInfo)
    end function

    function hipsolverZunmtr_rank_1(handle,side,uplo,trans,m,n,A,lda,tau,C,ldc,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZunmtr_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_SIDE_LEFT)) :: side
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      complex(c_double_complex),target,dimension(:) :: C
      integer(c_int) :: ldc
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZunmtr_rank_1 = hipsolverZunmtr_(handle,side,uplo,trans,m,n,c_loc(A),lda,tau,c_loc(C),ldc,work,lwork,devInfo)
    end function

    function hipsolverSgebrd_full_rank(handle,m,n,A,lda,D,E,tauq,taup,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgebrd_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:) :: D
      real(c_float),target,dimension(:) :: E
      real(c_float),target,dimension(:) :: tauq
      real(c_float),target,dimension(:) :: taup
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSgebrd_full_rank = hipsolverSgebrd_(handle,m,n,c_loc(A),lda,c_loc(D),c_loc(E),c_loc(tauq),c_loc(taup),work,lwork,devInfo)
    end function

    function hipsolverSgebrd_rank_0(handle,m,n,A,lda,D,E,tauq,taup,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgebrd_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_float),target :: A
      integer(c_int) :: lda
      real(c_float),target :: D
      real(c_float),target :: E
      real(c_float),target :: tauq
      real(c_float),target :: taup
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSgebrd_rank_0 = hipsolverSgebrd_(handle,m,n,c_loc(A),lda,c_loc(D),c_loc(E),c_loc(tauq),c_loc(taup),work,lwork,devInfo)
    end function

    function hipsolverSgebrd_rank_1(handle,m,n,A,lda,D,E,tauq,taup,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgebrd_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:) :: D
      real(c_float),target,dimension(:) :: E
      real(c_float),target,dimension(:) :: tauq
      real(c_float),target,dimension(:) :: taup
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSgebrd_rank_1 = hipsolverSgebrd_(handle,m,n,c_loc(A),lda,c_loc(D),c_loc(E),c_loc(tauq),c_loc(taup),work,lwork,devInfo)
    end function

    function hipsolverDgebrd_full_rank(handle,m,n,A,lda,D,E,tauq,taup,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgebrd_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:) :: D
      real(c_double),target,dimension(:) :: E
      real(c_double),target,dimension(:) :: tauq
      real(c_double),target,dimension(:) :: taup
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDgebrd_full_rank = hipsolverDgebrd_(handle,m,n,c_loc(A),lda,c_loc(D),c_loc(E),c_loc(tauq),c_loc(taup),work,lwork,devInfo)
    end function

    function hipsolverDgebrd_rank_0(handle,m,n,A,lda,D,E,tauq,taup,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgebrd_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_double),target :: A
      integer(c_int) :: lda
      real(c_double),target :: D
      real(c_double),target :: E
      real(c_double),target :: tauq
      real(c_double),target :: taup
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDgebrd_rank_0 = hipsolverDgebrd_(handle,m,n,c_loc(A),lda,c_loc(D),c_loc(E),c_loc(tauq),c_loc(taup),work,lwork,devInfo)
    end function

    function hipsolverDgebrd_rank_1(handle,m,n,A,lda,D,E,tauq,taup,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgebrd_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:) :: D
      real(c_double),target,dimension(:) :: E
      real(c_double),target,dimension(:) :: tauq
      real(c_double),target,dimension(:) :: taup
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDgebrd_rank_1 = hipsolverDgebrd_(handle,m,n,c_loc(A),lda,c_loc(D),c_loc(E),c_loc(tauq),c_loc(taup),work,lwork,devInfo)
    end function

    function hipsolverCgebrd_full_rank(handle,m,n,A,lda,D,E,tauq,taup,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgebrd_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:) :: D
      real(c_float),target,dimension(:) :: E
      complex(c_float_complex),target,dimension(:) :: tauq
      complex(c_float_complex),target,dimension(:) :: taup
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCgebrd_full_rank = hipsolverCgebrd_(handle,m,n,c_loc(A),lda,c_loc(D),c_loc(E),c_loc(tauq),c_loc(taup),work,lwork,devInfo)
    end function

    function hipsolverCgebrd_rank_0(handle,m,n,A,lda,D,E,tauq,taup,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgebrd_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      real(c_float),target :: D
      real(c_float),target :: E
      complex(c_float_complex),target :: tauq
      complex(c_float_complex),target :: taup
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCgebrd_rank_0 = hipsolverCgebrd_(handle,m,n,c_loc(A),lda,c_loc(D),c_loc(E),c_loc(tauq),c_loc(taup),work,lwork,devInfo)
    end function

    function hipsolverCgebrd_rank_1(handle,m,n,A,lda,D,E,tauq,taup,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgebrd_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:) :: D
      real(c_float),target,dimension(:) :: E
      complex(c_float_complex),target,dimension(:) :: tauq
      complex(c_float_complex),target,dimension(:) :: taup
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCgebrd_rank_1 = hipsolverCgebrd_(handle,m,n,c_loc(A),lda,c_loc(D),c_loc(E),c_loc(tauq),c_loc(taup),work,lwork,devInfo)
    end function

    function hipsolverZgebrd_full_rank(handle,m,n,A,lda,D,E,tauq,taup,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgebrd_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:) :: D
      real(c_double),target,dimension(:) :: E
      complex(c_double_complex),target,dimension(:) :: tauq
      complex(c_double_complex),target,dimension(:) :: taup
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZgebrd_full_rank = hipsolverZgebrd_(handle,m,n,c_loc(A),lda,c_loc(D),c_loc(E),c_loc(tauq),c_loc(taup),work,lwork,devInfo)
    end function

    function hipsolverZgebrd_rank_0(handle,m,n,A,lda,D,E,tauq,taup,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgebrd_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      real(c_double),target :: D
      real(c_double),target :: E
      complex(c_double_complex),target :: tauq
      complex(c_double_complex),target :: taup
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZgebrd_rank_0 = hipsolverZgebrd_(handle,m,n,c_loc(A),lda,c_loc(D),c_loc(E),c_loc(tauq),c_loc(taup),work,lwork,devInfo)
    end function

    function hipsolverZgebrd_rank_1(handle,m,n,A,lda,D,E,tauq,taup,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgebrd_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:) :: D
      real(c_double),target,dimension(:) :: E
      complex(c_double_complex),target,dimension(:) :: tauq
      complex(c_double_complex),target,dimension(:) :: taup
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZgebrd_rank_1 = hipsolverZgebrd_(handle,m,n,c_loc(A),lda,c_loc(D),c_loc(E),c_loc(tauq),c_loc(taup),work,lwork,devInfo)
    end function

    function hipsolverSgeqrf_bufferSize_full_rank(handle,m,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgeqrf_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverSgeqrf_bufferSize_full_rank = hipsolverSgeqrf_bufferSize_(handle,m,n,c_loc(A),lda,lwork)
    end function

    function hipsolverSgeqrf_bufferSize_rank_0(handle,m,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgeqrf_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_float),target :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverSgeqrf_bufferSize_rank_0 = hipsolverSgeqrf_bufferSize_(handle,m,n,c_loc(A),lda,lwork)
    end function

    function hipsolverSgeqrf_bufferSize_rank_1(handle,m,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgeqrf_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverSgeqrf_bufferSize_rank_1 = hipsolverSgeqrf_bufferSize_(handle,m,n,c_loc(A),lda,lwork)
    end function

    function hipsolverDgeqrf_bufferSize_full_rank(handle,m,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgeqrf_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverDgeqrf_bufferSize_full_rank = hipsolverDgeqrf_bufferSize_(handle,m,n,c_loc(A),lda,lwork)
    end function

    function hipsolverDgeqrf_bufferSize_rank_0(handle,m,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgeqrf_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_double),target :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverDgeqrf_bufferSize_rank_0 = hipsolverDgeqrf_bufferSize_(handle,m,n,c_loc(A),lda,lwork)
    end function

    function hipsolverDgeqrf_bufferSize_rank_1(handle,m,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgeqrf_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverDgeqrf_bufferSize_rank_1 = hipsolverDgeqrf_bufferSize_(handle,m,n,c_loc(A),lda,lwork)
    end function

    function hipsolverCgeqrf_bufferSize_full_rank(handle,m,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgeqrf_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverCgeqrf_bufferSize_full_rank = hipsolverCgeqrf_bufferSize_(handle,m,n,c_loc(A),lda,lwork)
    end function

    function hipsolverCgeqrf_bufferSize_rank_0(handle,m,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgeqrf_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverCgeqrf_bufferSize_rank_0 = hipsolverCgeqrf_bufferSize_(handle,m,n,c_loc(A),lda,lwork)
    end function

    function hipsolverCgeqrf_bufferSize_rank_1(handle,m,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgeqrf_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverCgeqrf_bufferSize_rank_1 = hipsolverCgeqrf_bufferSize_(handle,m,n,c_loc(A),lda,lwork)
    end function

    function hipsolverZgeqrf_bufferSize_full_rank(handle,m,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgeqrf_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverZgeqrf_bufferSize_full_rank = hipsolverZgeqrf_bufferSize_(handle,m,n,c_loc(A),lda,lwork)
    end function

    function hipsolverZgeqrf_bufferSize_rank_0(handle,m,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgeqrf_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverZgeqrf_bufferSize_rank_0 = hipsolverZgeqrf_bufferSize_(handle,m,n,c_loc(A),lda,lwork)
    end function

    function hipsolverZgeqrf_bufferSize_rank_1(handle,m,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgeqrf_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverZgeqrf_bufferSize_rank_1 = hipsolverZgeqrf_bufferSize_(handle,m,n,c_loc(A),lda,lwork)
    end function

    function hipsolverSgeqrf_full_rank(handle,m,n,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgeqrf_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSgeqrf_full_rank = hipsolverSgeqrf_(handle,m,n,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverSgeqrf_rank_0(handle,m,n,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgeqrf_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_float),target :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSgeqrf_rank_0 = hipsolverSgeqrf_(handle,m,n,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverSgeqrf_rank_1(handle,m,n,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgeqrf_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_float) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSgeqrf_rank_1 = hipsolverSgeqrf_(handle,m,n,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverDgeqrf_full_rank(handle,m,n,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgeqrf_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDgeqrf_full_rank = hipsolverDgeqrf_(handle,m,n,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverDgeqrf_rank_0(handle,m,n,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgeqrf_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_double),target :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDgeqrf_rank_0 = hipsolverDgeqrf_(handle,m,n,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverDgeqrf_rank_1(handle,m,n,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgeqrf_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_double) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDgeqrf_rank_1 = hipsolverDgeqrf_(handle,m,n,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverCgeqrf_full_rank(handle,m,n,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgeqrf_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCgeqrf_full_rank = hipsolverCgeqrf_(handle,m,n,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverCgeqrf_rank_0(handle,m,n,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgeqrf_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCgeqrf_rank_0 = hipsolverCgeqrf_(handle,m,n,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverCgeqrf_rank_1(handle,m,n,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgeqrf_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_float_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCgeqrf_rank_1 = hipsolverCgeqrf_(handle,m,n,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverZgeqrf_full_rank(handle,m,n,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgeqrf_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZgeqrf_full_rank = hipsolverZgeqrf_(handle,m,n,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverZgeqrf_rank_0(handle,m,n,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgeqrf_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZgeqrf_rank_0 = hipsolverZgeqrf_(handle,m,n,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverZgeqrf_rank_1(handle,m,n,A,lda,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgeqrf_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_double_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZgeqrf_rank_1 = hipsolverZgeqrf_(handle,m,n,c_loc(A),lda,tau,work,lwork,devInfo)
    end function

    function hipsolverSSgesv_bufferSize_full_rank(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSSgesv_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      real(c_float),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      real(c_float),target,dimension(:,:) :: X
      integer(c_int) :: ldx
      integer(c_size_t) :: lwork
      !
      hipsolverSSgesv_bufferSize_full_rank = hipsolverSSgesv_bufferSize_(handle,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,c_loc(X),ldx,lwork)
    end function

    function hipsolverSSgesv_bufferSize_rank_0(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSSgesv_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_float),target :: A
      integer(c_int) :: lda
      integer(c_int),target :: devIpiv
      real(c_float),target :: B
      integer(c_int) :: ldb
      real(c_float),target :: X
      integer(c_int) :: ldx
      integer(c_size_t) :: lwork
      !
      hipsolverSSgesv_bufferSize_rank_0 = hipsolverSSgesv_bufferSize_(handle,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,c_loc(X),ldx,lwork)
    end function

    function hipsolverSSgesv_bufferSize_rank_1(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSSgesv_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      real(c_float),target,dimension(:) :: B
      integer(c_int) :: ldb
      real(c_float),target,dimension(:) :: X
      integer(c_int) :: ldx
      integer(c_size_t) :: lwork
      !
      hipsolverSSgesv_bufferSize_rank_1 = hipsolverSSgesv_bufferSize_(handle,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,c_loc(X),ldx,lwork)
    end function

    function hipsolverDDgesv_bufferSize_full_rank(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDDgesv_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      real(c_double),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      real(c_double),target,dimension(:,:) :: X
      integer(c_int) :: ldx
      integer(c_size_t) :: lwork
      !
      hipsolverDDgesv_bufferSize_full_rank = hipsolverDDgesv_bufferSize_(handle,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,c_loc(X),ldx,lwork)
    end function

    function hipsolverDDgesv_bufferSize_rank_0(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDDgesv_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_double),target :: A
      integer(c_int) :: lda
      integer(c_int),target :: devIpiv
      real(c_double),target :: B
      integer(c_int) :: ldb
      real(c_double),target :: X
      integer(c_int) :: ldx
      integer(c_size_t) :: lwork
      !
      hipsolverDDgesv_bufferSize_rank_0 = hipsolverDDgesv_bufferSize_(handle,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,c_loc(X),ldx,lwork)
    end function

    function hipsolverDDgesv_bufferSize_rank_1(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDDgesv_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      real(c_double),target,dimension(:) :: B
      integer(c_int) :: ldb
      real(c_double),target,dimension(:) :: X
      integer(c_int) :: ldx
      integer(c_size_t) :: lwork
      !
      hipsolverDDgesv_bufferSize_rank_1 = hipsolverDDgesv_bufferSize_(handle,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,c_loc(X),ldx,lwork)
    end function

    function hipsolverCCgesv_bufferSize_full_rank(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCCgesv_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      complex(c_float_complex),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      complex(c_float_complex),target,dimension(:,:) :: X
      integer(c_int) :: ldx
      integer(c_size_t) :: lwork
      !
      hipsolverCCgesv_bufferSize_full_rank = hipsolverCCgesv_bufferSize_(handle,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,c_loc(X),ldx,lwork)
    end function

    function hipsolverCCgesv_bufferSize_rank_0(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCCgesv_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      integer(c_int),target :: devIpiv
      complex(c_float_complex),target :: B
      integer(c_int) :: ldb
      complex(c_float_complex),target :: X
      integer(c_int) :: ldx
      integer(c_size_t) :: lwork
      !
      hipsolverCCgesv_bufferSize_rank_0 = hipsolverCCgesv_bufferSize_(handle,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,c_loc(X),ldx,lwork)
    end function

    function hipsolverCCgesv_bufferSize_rank_1(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCCgesv_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      complex(c_float_complex),target,dimension(:) :: B
      integer(c_int) :: ldb
      complex(c_float_complex),target,dimension(:) :: X
      integer(c_int) :: ldx
      integer(c_size_t) :: lwork
      !
      hipsolverCCgesv_bufferSize_rank_1 = hipsolverCCgesv_bufferSize_(handle,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,c_loc(X),ldx,lwork)
    end function

    function hipsolverZZgesv_bufferSize_full_rank(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZZgesv_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      complex(c_double_complex),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      complex(c_double_complex),target,dimension(:,:) :: X
      integer(c_int) :: ldx
      integer(c_size_t) :: lwork
      !
      hipsolverZZgesv_bufferSize_full_rank = hipsolverZZgesv_bufferSize_(handle,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,c_loc(X),ldx,lwork)
    end function

    function hipsolverZZgesv_bufferSize_rank_0(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZZgesv_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      integer(c_int),target :: devIpiv
      complex(c_double_complex),target :: B
      integer(c_int) :: ldb
      complex(c_double_complex),target :: X
      integer(c_int) :: ldx
      integer(c_size_t) :: lwork
      !
      hipsolverZZgesv_bufferSize_rank_0 = hipsolverZZgesv_bufferSize_(handle,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,c_loc(X),ldx,lwork)
    end function

    function hipsolverZZgesv_bufferSize_rank_1(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZZgesv_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      complex(c_double_complex),target,dimension(:) :: B
      integer(c_int) :: ldb
      complex(c_double_complex),target,dimension(:) :: X
      integer(c_int) :: ldx
      integer(c_size_t) :: lwork
      !
      hipsolverZZgesv_bufferSize_rank_1 = hipsolverZZgesv_bufferSize_(handle,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,c_loc(X),ldx,lwork)
    end function

    function hipsolverSSgesv_full_rank(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,work,lwork,niters,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSSgesv_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      real(c_float),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      real(c_float),target,dimension(:,:) :: X
      integer(c_int) :: ldx
      type(c_ptr) :: work
      integer(c_size_t) :: lwork
      type(c_ptr) :: niters
      integer(c_int) :: devInfo
      !
      hipsolverSSgesv_full_rank = hipsolverSSgesv_(handle,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,c_loc(X),ldx,work,lwork,niters,devInfo)
    end function

    function hipsolverSSgesv_rank_0(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,work,lwork,niters,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSSgesv_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_float),target :: A
      integer(c_int) :: lda
      integer(c_int),target :: devIpiv
      real(c_float),target :: B
      integer(c_int) :: ldb
      real(c_float),target :: X
      integer(c_int) :: ldx
      type(c_ptr) :: work
      integer(c_size_t) :: lwork
      type(c_ptr) :: niters
      integer(c_int) :: devInfo
      !
      hipsolverSSgesv_rank_0 = hipsolverSSgesv_(handle,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,c_loc(X),ldx,work,lwork,niters,devInfo)
    end function

    function hipsolverSSgesv_rank_1(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,work,lwork,niters,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSSgesv_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      real(c_float),target,dimension(:) :: B
      integer(c_int) :: ldb
      real(c_float),target,dimension(:) :: X
      integer(c_int) :: ldx
      type(c_ptr) :: work
      integer(c_size_t) :: lwork
      type(c_ptr) :: niters
      integer(c_int) :: devInfo
      !
      hipsolverSSgesv_rank_1 = hipsolverSSgesv_(handle,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,c_loc(X),ldx,work,lwork,niters,devInfo)
    end function

    function hipsolverDDgesv_full_rank(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,work,lwork,niters,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDDgesv_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      real(c_double),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      real(c_double),target,dimension(:,:) :: X
      integer(c_int) :: ldx
      type(c_ptr) :: work
      integer(c_size_t) :: lwork
      type(c_ptr) :: niters
      integer(c_int) :: devInfo
      !
      hipsolverDDgesv_full_rank = hipsolverDDgesv_(handle,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,c_loc(X),ldx,work,lwork,niters,devInfo)
    end function

    function hipsolverDDgesv_rank_0(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,work,lwork,niters,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDDgesv_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_double),target :: A
      integer(c_int) :: lda
      integer(c_int),target :: devIpiv
      real(c_double),target :: B
      integer(c_int) :: ldb
      real(c_double),target :: X
      integer(c_int) :: ldx
      type(c_ptr) :: work
      integer(c_size_t) :: lwork
      type(c_ptr) :: niters
      integer(c_int) :: devInfo
      !
      hipsolverDDgesv_rank_0 = hipsolverDDgesv_(handle,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,c_loc(X),ldx,work,lwork,niters,devInfo)
    end function

    function hipsolverDDgesv_rank_1(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,work,lwork,niters,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDDgesv_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      real(c_double),target,dimension(:) :: B
      integer(c_int) :: ldb
      real(c_double),target,dimension(:) :: X
      integer(c_int) :: ldx
      type(c_ptr) :: work
      integer(c_size_t) :: lwork
      type(c_ptr) :: niters
      integer(c_int) :: devInfo
      !
      hipsolverDDgesv_rank_1 = hipsolverDDgesv_(handle,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,c_loc(X),ldx,work,lwork,niters,devInfo)
    end function

    function hipsolverCCgesv_full_rank(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,work,lwork,niters,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCCgesv_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      complex(c_float_complex),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      complex(c_float_complex),target,dimension(:,:) :: X
      integer(c_int) :: ldx
      type(c_ptr) :: work
      integer(c_size_t) :: lwork
      type(c_ptr) :: niters
      integer(c_int) :: devInfo
      !
      hipsolverCCgesv_full_rank = hipsolverCCgesv_(handle,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,c_loc(X),ldx,work,lwork,niters,devInfo)
    end function

    function hipsolverCCgesv_rank_0(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,work,lwork,niters,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCCgesv_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      integer(c_int),target :: devIpiv
      complex(c_float_complex),target :: B
      integer(c_int) :: ldb
      complex(c_float_complex),target :: X
      integer(c_int) :: ldx
      type(c_ptr) :: work
      integer(c_size_t) :: lwork
      type(c_ptr) :: niters
      integer(c_int) :: devInfo
      !
      hipsolverCCgesv_rank_0 = hipsolverCCgesv_(handle,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,c_loc(X),ldx,work,lwork,niters,devInfo)
    end function

    function hipsolverCCgesv_rank_1(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,work,lwork,niters,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCCgesv_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      complex(c_float_complex),target,dimension(:) :: B
      integer(c_int) :: ldb
      complex(c_float_complex),target,dimension(:) :: X
      integer(c_int) :: ldx
      type(c_ptr) :: work
      integer(c_size_t) :: lwork
      type(c_ptr) :: niters
      integer(c_int) :: devInfo
      !
      hipsolverCCgesv_rank_1 = hipsolverCCgesv_(handle,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,c_loc(X),ldx,work,lwork,niters,devInfo)
    end function

    function hipsolverZZgesv_full_rank(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,work,lwork,niters,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZZgesv_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      complex(c_double_complex),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      complex(c_double_complex),target,dimension(:,:) :: X
      integer(c_int) :: ldx
      type(c_ptr) :: work
      integer(c_size_t) :: lwork
      type(c_ptr) :: niters
      integer(c_int) :: devInfo
      !
      hipsolverZZgesv_full_rank = hipsolverZZgesv_(handle,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,c_loc(X),ldx,work,lwork,niters,devInfo)
    end function

    function hipsolverZZgesv_rank_0(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,work,lwork,niters,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZZgesv_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      integer(c_int),target :: devIpiv
      complex(c_double_complex),target :: B
      integer(c_int) :: ldb
      complex(c_double_complex),target :: X
      integer(c_int) :: ldx
      type(c_ptr) :: work
      integer(c_size_t) :: lwork
      type(c_ptr) :: niters
      integer(c_int) :: devInfo
      !
      hipsolverZZgesv_rank_0 = hipsolverZZgesv_(handle,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,c_loc(X),ldx,work,lwork,niters,devInfo)
    end function

    function hipsolverZZgesv_rank_1(handle,n,nrhs,A,lda,devIpiv,B,ldb,X,ldx,work,lwork,niters,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZZgesv_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      complex(c_double_complex),target,dimension(:) :: B
      integer(c_int) :: ldb
      complex(c_double_complex),target,dimension(:) :: X
      integer(c_int) :: ldx
      type(c_ptr) :: work
      integer(c_size_t) :: lwork
      type(c_ptr) :: niters
      integer(c_int) :: devInfo
      !
      hipsolverZZgesv_rank_1 = hipsolverZZgesv_(handle,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,c_loc(X),ldx,work,lwork,niters,devInfo)
    end function

    function hipsolverSgetrf_bufferSize_full_rank(handle,m,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgetrf_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverSgetrf_bufferSize_full_rank = hipsolverSgetrf_bufferSize_(handle,m,n,c_loc(A),lda,lwork)
    end function

    function hipsolverSgetrf_bufferSize_rank_0(handle,m,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgetrf_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_float),target :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverSgetrf_bufferSize_rank_0 = hipsolverSgetrf_bufferSize_(handle,m,n,c_loc(A),lda,lwork)
    end function

    function hipsolverSgetrf_bufferSize_rank_1(handle,m,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgetrf_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverSgetrf_bufferSize_rank_1 = hipsolverSgetrf_bufferSize_(handle,m,n,c_loc(A),lda,lwork)
    end function

    function hipsolverDgetrf_bufferSize_full_rank(handle,m,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgetrf_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverDgetrf_bufferSize_full_rank = hipsolverDgetrf_bufferSize_(handle,m,n,c_loc(A),lda,lwork)
    end function

    function hipsolverDgetrf_bufferSize_rank_0(handle,m,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgetrf_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_double),target :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverDgetrf_bufferSize_rank_0 = hipsolverDgetrf_bufferSize_(handle,m,n,c_loc(A),lda,lwork)
    end function

    function hipsolverDgetrf_bufferSize_rank_1(handle,m,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgetrf_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverDgetrf_bufferSize_rank_1 = hipsolverDgetrf_bufferSize_(handle,m,n,c_loc(A),lda,lwork)
    end function

    function hipsolverCgetrf_bufferSize_full_rank(handle,m,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgetrf_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverCgetrf_bufferSize_full_rank = hipsolverCgetrf_bufferSize_(handle,m,n,c_loc(A),lda,lwork)
    end function

    function hipsolverCgetrf_bufferSize_rank_0(handle,m,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgetrf_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverCgetrf_bufferSize_rank_0 = hipsolverCgetrf_bufferSize_(handle,m,n,c_loc(A),lda,lwork)
    end function

    function hipsolverCgetrf_bufferSize_rank_1(handle,m,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgetrf_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverCgetrf_bufferSize_rank_1 = hipsolverCgetrf_bufferSize_(handle,m,n,c_loc(A),lda,lwork)
    end function

    function hipsolverZgetrf_bufferSize_full_rank(handle,m,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgetrf_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverZgetrf_bufferSize_full_rank = hipsolverZgetrf_bufferSize_(handle,m,n,c_loc(A),lda,lwork)
    end function

    function hipsolverZgetrf_bufferSize_rank_0(handle,m,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgetrf_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverZgetrf_bufferSize_rank_0 = hipsolverZgetrf_bufferSize_(handle,m,n,c_loc(A),lda,lwork)
    end function

    function hipsolverZgetrf_bufferSize_rank_1(handle,m,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgetrf_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverZgetrf_bufferSize_rank_1 = hipsolverZgetrf_bufferSize_(handle,m,n,c_loc(A),lda,lwork)
    end function

    function hipsolverSgetrf_full_rank(handle,m,n,A,lda,work,lwork,devIpiv,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgetrf_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int),target,dimension(:) :: devIpiv
      integer(c_int) :: devInfo
      !
      hipsolverSgetrf_full_rank = hipsolverSgetrf_(handle,m,n,c_loc(A),lda,work,lwork,c_loc(devIpiv),devInfo)
    end function

    function hipsolverSgetrf_rank_0(handle,m,n,A,lda,work,lwork,devIpiv,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgetrf_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_float),target :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int),target :: devIpiv
      integer(c_int) :: devInfo
      !
      hipsolverSgetrf_rank_0 = hipsolverSgetrf_(handle,m,n,c_loc(A),lda,work,lwork,c_loc(devIpiv),devInfo)
    end function

    function hipsolverSgetrf_rank_1(handle,m,n,A,lda,work,lwork,devIpiv,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgetrf_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int),target,dimension(:) :: devIpiv
      integer(c_int) :: devInfo
      !
      hipsolverSgetrf_rank_1 = hipsolverSgetrf_(handle,m,n,c_loc(A),lda,work,lwork,c_loc(devIpiv),devInfo)
    end function

    function hipsolverDgetrf_full_rank(handle,m,n,A,lda,work,lwork,devIpiv,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgetrf_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int),target,dimension(:) :: devIpiv
      integer(c_int) :: devInfo
      !
      hipsolverDgetrf_full_rank = hipsolverDgetrf_(handle,m,n,c_loc(A),lda,work,lwork,c_loc(devIpiv),devInfo)
    end function

    function hipsolverDgetrf_rank_0(handle,m,n,A,lda,work,lwork,devIpiv,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgetrf_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_double),target :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int),target :: devIpiv
      integer(c_int) :: devInfo
      !
      hipsolverDgetrf_rank_0 = hipsolverDgetrf_(handle,m,n,c_loc(A),lda,work,lwork,c_loc(devIpiv),devInfo)
    end function

    function hipsolverDgetrf_rank_1(handle,m,n,A,lda,work,lwork,devIpiv,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgetrf_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int),target,dimension(:) :: devIpiv
      integer(c_int) :: devInfo
      !
      hipsolverDgetrf_rank_1 = hipsolverDgetrf_(handle,m,n,c_loc(A),lda,work,lwork,c_loc(devIpiv),devInfo)
    end function

    function hipsolverCgetrf_full_rank(handle,m,n,A,lda,work,lwork,devIpiv,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgetrf_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int),target,dimension(:) :: devIpiv
      integer(c_int) :: devInfo
      !
      hipsolverCgetrf_full_rank = hipsolverCgetrf_(handle,m,n,c_loc(A),lda,work,lwork,c_loc(devIpiv),devInfo)
    end function

    function hipsolverCgetrf_rank_0(handle,m,n,A,lda,work,lwork,devIpiv,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgetrf_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int),target :: devIpiv
      integer(c_int) :: devInfo
      !
      hipsolverCgetrf_rank_0 = hipsolverCgetrf_(handle,m,n,c_loc(A),lda,work,lwork,c_loc(devIpiv),devInfo)
    end function

    function hipsolverCgetrf_rank_1(handle,m,n,A,lda,work,lwork,devIpiv,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgetrf_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int),target,dimension(:) :: devIpiv
      integer(c_int) :: devInfo
      !
      hipsolverCgetrf_rank_1 = hipsolverCgetrf_(handle,m,n,c_loc(A),lda,work,lwork,c_loc(devIpiv),devInfo)
    end function

    function hipsolverZgetrf_full_rank(handle,m,n,A,lda,work,lwork,devIpiv,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgetrf_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int),target,dimension(:) :: devIpiv
      integer(c_int) :: devInfo
      !
      hipsolverZgetrf_full_rank = hipsolverZgetrf_(handle,m,n,c_loc(A),lda,work,lwork,c_loc(devIpiv),devInfo)
    end function

    function hipsolverZgetrf_rank_0(handle,m,n,A,lda,work,lwork,devIpiv,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgetrf_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int),target :: devIpiv
      integer(c_int) :: devInfo
      !
      hipsolverZgetrf_rank_0 = hipsolverZgetrf_(handle,m,n,c_loc(A),lda,work,lwork,c_loc(devIpiv),devInfo)
    end function

    function hipsolverZgetrf_rank_1(handle,m,n,A,lda,work,lwork,devIpiv,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgetrf_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: m
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int),target,dimension(:) :: devIpiv
      integer(c_int) :: devInfo
      !
      hipsolverZgetrf_rank_1 = hipsolverZgetrf_(handle,m,n,c_loc(A),lda,work,lwork,c_loc(devIpiv),devInfo)
    end function

    function hipsolverSgetrs_bufferSize_full_rank(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgetrs_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      real(c_float),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      !
      hipsolverSgetrs_bufferSize_full_rank = hipsolverSgetrs_bufferSize_(handle,trans,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,lwork)
    end function

    function hipsolverSgetrs_bufferSize_rank_0(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgetrs_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_float),target :: A
      integer(c_int) :: lda
      integer(c_int),target :: devIpiv
      real(c_float),target :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      !
      hipsolverSgetrs_bufferSize_rank_0 = hipsolverSgetrs_bufferSize_(handle,trans,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,lwork)
    end function

    function hipsolverSgetrs_bufferSize_rank_1(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgetrs_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      real(c_float),target,dimension(:) :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      !
      hipsolverSgetrs_bufferSize_rank_1 = hipsolverSgetrs_bufferSize_(handle,trans,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,lwork)
    end function

    function hipsolverDgetrs_bufferSize_full_rank(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgetrs_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      real(c_double),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      !
      hipsolverDgetrs_bufferSize_full_rank = hipsolverDgetrs_bufferSize_(handle,trans,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,lwork)
    end function

    function hipsolverDgetrs_bufferSize_rank_0(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgetrs_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_double),target :: A
      integer(c_int) :: lda
      integer(c_int),target :: devIpiv
      real(c_double),target :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      !
      hipsolverDgetrs_bufferSize_rank_0 = hipsolverDgetrs_bufferSize_(handle,trans,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,lwork)
    end function

    function hipsolverDgetrs_bufferSize_rank_1(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgetrs_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      real(c_double),target,dimension(:) :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      !
      hipsolverDgetrs_bufferSize_rank_1 = hipsolverDgetrs_bufferSize_(handle,trans,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,lwork)
    end function

    function hipsolverCgetrs_bufferSize_full_rank(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgetrs_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      complex(c_float_complex),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      !
      hipsolverCgetrs_bufferSize_full_rank = hipsolverCgetrs_bufferSize_(handle,trans,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,lwork)
    end function

    function hipsolverCgetrs_bufferSize_rank_0(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgetrs_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      integer(c_int),target :: devIpiv
      complex(c_float_complex),target :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      !
      hipsolverCgetrs_bufferSize_rank_0 = hipsolverCgetrs_bufferSize_(handle,trans,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,lwork)
    end function

    function hipsolverCgetrs_bufferSize_rank_1(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgetrs_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      complex(c_float_complex),target,dimension(:) :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      !
      hipsolverCgetrs_bufferSize_rank_1 = hipsolverCgetrs_bufferSize_(handle,trans,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,lwork)
    end function

    function hipsolverZgetrs_bufferSize_full_rank(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgetrs_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      complex(c_double_complex),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      !
      hipsolverZgetrs_bufferSize_full_rank = hipsolverZgetrs_bufferSize_(handle,trans,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,lwork)
    end function

    function hipsolverZgetrs_bufferSize_rank_0(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgetrs_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      integer(c_int),target :: devIpiv
      complex(c_double_complex),target :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      !
      hipsolverZgetrs_bufferSize_rank_0 = hipsolverZgetrs_bufferSize_(handle,trans,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,lwork)
    end function

    function hipsolverZgetrs_bufferSize_rank_1(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgetrs_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      complex(c_double_complex),target,dimension(:) :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      !
      hipsolverZgetrs_bufferSize_rank_1 = hipsolverZgetrs_bufferSize_(handle,trans,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,lwork)
    end function

    function hipsolverSgetrs_full_rank(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgetrs_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      real(c_float),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSgetrs_full_rank = hipsolverSgetrs_(handle,trans,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,work,lwork,devInfo)
    end function

    function hipsolverSgetrs_rank_0(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgetrs_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_float),target :: A
      integer(c_int) :: lda
      integer(c_int),target :: devIpiv
      real(c_float),target :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSgetrs_rank_0 = hipsolverSgetrs_(handle,trans,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,work,lwork,devInfo)
    end function

    function hipsolverSgetrs_rank_1(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgetrs_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      real(c_float),target,dimension(:) :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSgetrs_rank_1 = hipsolverSgetrs_(handle,trans,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,work,lwork,devInfo)
    end function

    function hipsolverDgetrs_full_rank(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgetrs_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      real(c_double),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDgetrs_full_rank = hipsolverDgetrs_(handle,trans,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,work,lwork,devInfo)
    end function

    function hipsolverDgetrs_rank_0(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgetrs_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_double),target :: A
      integer(c_int) :: lda
      integer(c_int),target :: devIpiv
      real(c_double),target :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDgetrs_rank_0 = hipsolverDgetrs_(handle,trans,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,work,lwork,devInfo)
    end function

    function hipsolverDgetrs_rank_1(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgetrs_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      real(c_double),target,dimension(:) :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDgetrs_rank_1 = hipsolverDgetrs_(handle,trans,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,work,lwork,devInfo)
    end function

    function hipsolverCgetrs_full_rank(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgetrs_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      complex(c_float_complex),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCgetrs_full_rank = hipsolverCgetrs_(handle,trans,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,work,lwork,devInfo)
    end function

    function hipsolverCgetrs_rank_0(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgetrs_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      integer(c_int),target :: devIpiv
      complex(c_float_complex),target :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCgetrs_rank_0 = hipsolverCgetrs_(handle,trans,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,work,lwork,devInfo)
    end function

    function hipsolverCgetrs_rank_1(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgetrs_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      complex(c_float_complex),target,dimension(:) :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCgetrs_rank_1 = hipsolverCgetrs_(handle,trans,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,work,lwork,devInfo)
    end function

    function hipsolverZgetrs_full_rank(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgetrs_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      complex(c_double_complex),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZgetrs_full_rank = hipsolverZgetrs_(handle,trans,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,work,lwork,devInfo)
    end function

    function hipsolverZgetrs_rank_0(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgetrs_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      integer(c_int),target :: devIpiv
      complex(c_double_complex),target :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZgetrs_rank_0 = hipsolverZgetrs_(handle,trans,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,work,lwork,devInfo)
    end function

    function hipsolverZgetrs_rank_1(handle,trans,n,nrhs,A,lda,devIpiv,B,ldb,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgetrs_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_OP_N)) :: trans
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int),target,dimension(:) :: devIpiv
      complex(c_double_complex),target,dimension(:) :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZgetrs_rank_1 = hipsolverZgetrs_(handle,trans,n,nrhs,c_loc(A),lda,c_loc(devIpiv),c_loc(B),ldb,work,lwork,devInfo)
    end function

    function hipsolverSpotrf_bufferSize_full_rank(handle,uplo,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrf_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverSpotrf_bufferSize_full_rank = hipsolverSpotrf_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork)
    end function

    function hipsolverSpotrf_bufferSize_rank_0(handle,uplo,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrf_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverSpotrf_bufferSize_rank_0 = hipsolverSpotrf_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork)
    end function

    function hipsolverSpotrf_bufferSize_rank_1(handle,uplo,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrf_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverSpotrf_bufferSize_rank_1 = hipsolverSpotrf_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork)
    end function

    function hipsolverDpotrf_bufferSize_full_rank(handle,uplo,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrf_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverDpotrf_bufferSize_full_rank = hipsolverDpotrf_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork)
    end function

    function hipsolverDpotrf_bufferSize_rank_0(handle,uplo,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrf_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverDpotrf_bufferSize_rank_0 = hipsolverDpotrf_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork)
    end function

    function hipsolverDpotrf_bufferSize_rank_1(handle,uplo,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrf_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverDpotrf_bufferSize_rank_1 = hipsolverDpotrf_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork)
    end function

    function hipsolverCpotrf_bufferSize_full_rank(handle,uplo,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrf_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverCpotrf_bufferSize_full_rank = hipsolverCpotrf_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork)
    end function

    function hipsolverCpotrf_bufferSize_rank_0(handle,uplo,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrf_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverCpotrf_bufferSize_rank_0 = hipsolverCpotrf_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork)
    end function

    function hipsolverCpotrf_bufferSize_rank_1(handle,uplo,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrf_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverCpotrf_bufferSize_rank_1 = hipsolverCpotrf_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork)
    end function

    function hipsolverZpotrf_bufferSize_full_rank(handle,uplo,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrf_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverZpotrf_bufferSize_full_rank = hipsolverZpotrf_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork)
    end function

    function hipsolverZpotrf_bufferSize_rank_0(handle,uplo,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrf_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverZpotrf_bufferSize_rank_0 = hipsolverZpotrf_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork)
    end function

    function hipsolverZpotrf_bufferSize_rank_1(handle,uplo,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrf_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverZpotrf_bufferSize_rank_1 = hipsolverZpotrf_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork)
    end function

    function hipsolverSpotrf_full_rank(handle,uplo,n,A,lda,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrf_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSpotrf_full_rank = hipsolverSpotrf_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo)
    end function

    function hipsolverSpotrf_rank_0(handle,uplo,n,A,lda,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrf_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSpotrf_rank_0 = hipsolverSpotrf_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo)
    end function

    function hipsolverSpotrf_rank_1(handle,uplo,n,A,lda,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrf_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSpotrf_rank_1 = hipsolverSpotrf_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo)
    end function

    function hipsolverDpotrf_full_rank(handle,uplo,n,A,lda,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrf_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDpotrf_full_rank = hipsolverDpotrf_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo)
    end function

    function hipsolverDpotrf_rank_0(handle,uplo,n,A,lda,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrf_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDpotrf_rank_0 = hipsolverDpotrf_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo)
    end function

    function hipsolverDpotrf_rank_1(handle,uplo,n,A,lda,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrf_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDpotrf_rank_1 = hipsolverDpotrf_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo)
    end function

    function hipsolverCpotrf_full_rank(handle,uplo,n,A,lda,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrf_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCpotrf_full_rank = hipsolverCpotrf_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo)
    end function

    function hipsolverCpotrf_rank_0(handle,uplo,n,A,lda,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrf_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCpotrf_rank_0 = hipsolverCpotrf_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo)
    end function

    function hipsolverCpotrf_rank_1(handle,uplo,n,A,lda,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrf_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCpotrf_rank_1 = hipsolverCpotrf_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo)
    end function

    function hipsolverZpotrf_full_rank(handle,uplo,n,A,lda,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrf_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZpotrf_full_rank = hipsolverZpotrf_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo)
    end function

    function hipsolverZpotrf_rank_0(handle,uplo,n,A,lda,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrf_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZpotrf_rank_0 = hipsolverZpotrf_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo)
    end function

    function hipsolverZpotrf_rank_1(handle,uplo,n,A,lda,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrf_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZpotrf_rank_1 = hipsolverZpotrf_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo)
    end function

    function hipsolverSpotrfBatched_bufferSize_full_rank(handle,uplo,n,A,lda,lwork,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrfBatched_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:,:,:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      integer(c_int) :: batch_count
      !
      hipsolverSpotrfBatched_bufferSize_full_rank = hipsolverSpotrfBatched_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork,batch_count)
    end function

    function hipsolverSpotrfBatched_bufferSize_rank_0(handle,uplo,n,A,lda,lwork,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrfBatched_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      integer(c_int) :: batch_count
      !
      hipsolverSpotrfBatched_bufferSize_rank_0 = hipsolverSpotrfBatched_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork,batch_count)
    end function

    function hipsolverSpotrfBatched_bufferSize_rank_1(handle,uplo,n,A,lda,lwork,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrfBatched_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      integer(c_int) :: batch_count
      !
      hipsolverSpotrfBatched_bufferSize_rank_1 = hipsolverSpotrfBatched_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork,batch_count)
    end function

    function hipsolverDpotrfBatched_bufferSize_full_rank(handle,uplo,n,A,lda,lwork,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrfBatched_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:,:,:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      integer(c_int) :: batch_count
      !
      hipsolverDpotrfBatched_bufferSize_full_rank = hipsolverDpotrfBatched_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork,batch_count)
    end function

    function hipsolverDpotrfBatched_bufferSize_rank_0(handle,uplo,n,A,lda,lwork,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrfBatched_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      integer(c_int) :: batch_count
      !
      hipsolverDpotrfBatched_bufferSize_rank_0 = hipsolverDpotrfBatched_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork,batch_count)
    end function

    function hipsolverDpotrfBatched_bufferSize_rank_1(handle,uplo,n,A,lda,lwork,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrfBatched_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      integer(c_int) :: batch_count
      !
      hipsolverDpotrfBatched_bufferSize_rank_1 = hipsolverDpotrfBatched_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork,batch_count)
    end function

    function hipsolverCpotrfBatched_bufferSize_full_rank(handle,uplo,n,A,lda,lwork,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrfBatched_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:,:,:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      integer(c_int) :: batch_count
      !
      hipsolverCpotrfBatched_bufferSize_full_rank = hipsolverCpotrfBatched_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork,batch_count)
    end function

    function hipsolverCpotrfBatched_bufferSize_rank_0(handle,uplo,n,A,lda,lwork,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrfBatched_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      integer(c_int) :: batch_count
      !
      hipsolverCpotrfBatched_bufferSize_rank_0 = hipsolverCpotrfBatched_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork,batch_count)
    end function

    function hipsolverCpotrfBatched_bufferSize_rank_1(handle,uplo,n,A,lda,lwork,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrfBatched_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      integer(c_int) :: batch_count
      !
      hipsolverCpotrfBatched_bufferSize_rank_1 = hipsolverCpotrfBatched_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork,batch_count)
    end function

    function hipsolverZpotrfBatched_bufferSize_full_rank(handle,uplo,n,A,lda,lwork,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrfBatched_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:,:,:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      integer(c_int) :: batch_count
      !
      hipsolverZpotrfBatched_bufferSize_full_rank = hipsolverZpotrfBatched_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork,batch_count)
    end function

    function hipsolverZpotrfBatched_bufferSize_rank_0(handle,uplo,n,A,lda,lwork,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrfBatched_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      integer(c_int) :: batch_count
      !
      hipsolverZpotrfBatched_bufferSize_rank_0 = hipsolverZpotrfBatched_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork,batch_count)
    end function

    function hipsolverZpotrfBatched_bufferSize_rank_1(handle,uplo,n,A,lda,lwork,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrfBatched_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      integer(c_int) :: batch_count
      !
      hipsolverZpotrfBatched_bufferSize_rank_1 = hipsolverZpotrfBatched_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork,batch_count)
    end function

    function hipsolverSpotrfBatched_full_rank(handle,uplo,n,A,lda,work,lwork,devInfo,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrfBatched_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:,:,:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      integer(c_int) :: batch_count
      !
      hipsolverSpotrfBatched_full_rank = hipsolverSpotrfBatched_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo,batch_count)
    end function

    function hipsolverSpotrfBatched_rank_0(handle,uplo,n,A,lda,work,lwork,devInfo,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrfBatched_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      integer(c_int) :: batch_count
      !
      hipsolverSpotrfBatched_rank_0 = hipsolverSpotrfBatched_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo,batch_count)
    end function

    function hipsolverSpotrfBatched_rank_1(handle,uplo,n,A,lda,work,lwork,devInfo,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrfBatched_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      integer(c_int) :: batch_count
      !
      hipsolverSpotrfBatched_rank_1 = hipsolverSpotrfBatched_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo,batch_count)
    end function

    function hipsolverDpotrfBatched_full_rank(handle,uplo,n,A,lda,work,lwork,devInfo,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrfBatched_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:,:,:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      integer(c_int) :: batch_count
      !
      hipsolverDpotrfBatched_full_rank = hipsolverDpotrfBatched_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo,batch_count)
    end function

    function hipsolverDpotrfBatched_rank_0(handle,uplo,n,A,lda,work,lwork,devInfo,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrfBatched_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      integer(c_int) :: batch_count
      !
      hipsolverDpotrfBatched_rank_0 = hipsolverDpotrfBatched_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo,batch_count)
    end function

    function hipsolverDpotrfBatched_rank_1(handle,uplo,n,A,lda,work,lwork,devInfo,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrfBatched_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      integer(c_int) :: batch_count
      !
      hipsolverDpotrfBatched_rank_1 = hipsolverDpotrfBatched_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo,batch_count)
    end function

    function hipsolverCpotrfBatched_full_rank(handle,uplo,n,A,lda,work,lwork,devInfo,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrfBatched_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:,:,:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      integer(c_int) :: batch_count
      !
      hipsolverCpotrfBatched_full_rank = hipsolverCpotrfBatched_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo,batch_count)
    end function

    function hipsolverCpotrfBatched_rank_0(handle,uplo,n,A,lda,work,lwork,devInfo,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrfBatched_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      integer(c_int) :: batch_count
      !
      hipsolverCpotrfBatched_rank_0 = hipsolverCpotrfBatched_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo,batch_count)
    end function

    function hipsolverCpotrfBatched_rank_1(handle,uplo,n,A,lda,work,lwork,devInfo,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrfBatched_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      integer(c_int) :: batch_count
      !
      hipsolverCpotrfBatched_rank_1 = hipsolverCpotrfBatched_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo,batch_count)
    end function

    function hipsolverZpotrfBatched_full_rank(handle,uplo,n,A,lda,work,lwork,devInfo,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrfBatched_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:,:,:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      integer(c_int) :: batch_count
      !
      hipsolverZpotrfBatched_full_rank = hipsolverZpotrfBatched_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo,batch_count)
    end function

    function hipsolverZpotrfBatched_rank_0(handle,uplo,n,A,lda,work,lwork,devInfo,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrfBatched_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      integer(c_int) :: batch_count
      !
      hipsolverZpotrfBatched_rank_0 = hipsolverZpotrfBatched_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo,batch_count)
    end function

    function hipsolverZpotrfBatched_rank_1(handle,uplo,n,A,lda,work,lwork,devInfo,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrfBatched_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      integer(c_int) :: batch_count
      !
      hipsolverZpotrfBatched_rank_1 = hipsolverZpotrfBatched_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo,batch_count)
    end function

    function hipsolverSpotri_bufferSize_full_rank(handle,uplo,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotri_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverSpotri_bufferSize_full_rank = hipsolverSpotri_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork)
    end function

    function hipsolverSpotri_bufferSize_rank_0(handle,uplo,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotri_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverSpotri_bufferSize_rank_0 = hipsolverSpotri_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork)
    end function

    function hipsolverSpotri_bufferSize_rank_1(handle,uplo,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotri_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverSpotri_bufferSize_rank_1 = hipsolverSpotri_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork)
    end function

    function hipsolverDpotri_bufferSize_full_rank(handle,uplo,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotri_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverDpotri_bufferSize_full_rank = hipsolverDpotri_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork)
    end function

    function hipsolverDpotri_bufferSize_rank_0(handle,uplo,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotri_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverDpotri_bufferSize_rank_0 = hipsolverDpotri_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork)
    end function

    function hipsolverDpotri_bufferSize_rank_1(handle,uplo,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotri_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverDpotri_bufferSize_rank_1 = hipsolverDpotri_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork)
    end function

    function hipsolverCpotri_bufferSize_full_rank(handle,uplo,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotri_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverCpotri_bufferSize_full_rank = hipsolverCpotri_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork)
    end function

    function hipsolverCpotri_bufferSize_rank_0(handle,uplo,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotri_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverCpotri_bufferSize_rank_0 = hipsolverCpotri_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork)
    end function

    function hipsolverCpotri_bufferSize_rank_1(handle,uplo,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotri_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverCpotri_bufferSize_rank_1 = hipsolverCpotri_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork)
    end function

    function hipsolverZpotri_bufferSize_full_rank(handle,uplo,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotri_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverZpotri_bufferSize_full_rank = hipsolverZpotri_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork)
    end function

    function hipsolverZpotri_bufferSize_rank_0(handle,uplo,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotri_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverZpotri_bufferSize_rank_0 = hipsolverZpotri_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork)
    end function

    function hipsolverZpotri_bufferSize_rank_1(handle,uplo,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotri_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverZpotri_bufferSize_rank_1 = hipsolverZpotri_bufferSize_(handle,uplo,n,c_loc(A),lda,lwork)
    end function

    function hipsolverSpotri_full_rank(handle,uplo,n,A,lda,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotri_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSpotri_full_rank = hipsolverSpotri_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo)
    end function

    function hipsolverSpotri_rank_0(handle,uplo,n,A,lda,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotri_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSpotri_rank_0 = hipsolverSpotri_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo)
    end function

    function hipsolverSpotri_rank_1(handle,uplo,n,A,lda,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotri_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSpotri_rank_1 = hipsolverSpotri_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo)
    end function

    function hipsolverDpotri_full_rank(handle,uplo,n,A,lda,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotri_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDpotri_full_rank = hipsolverDpotri_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo)
    end function

    function hipsolverDpotri_rank_0(handle,uplo,n,A,lda,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotri_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDpotri_rank_0 = hipsolverDpotri_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo)
    end function

    function hipsolverDpotri_rank_1(handle,uplo,n,A,lda,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotri_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDpotri_rank_1 = hipsolverDpotri_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo)
    end function

    function hipsolverCpotri_full_rank(handle,uplo,n,A,lda,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotri_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCpotri_full_rank = hipsolverCpotri_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo)
    end function

    function hipsolverCpotri_rank_0(handle,uplo,n,A,lda,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotri_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCpotri_rank_0 = hipsolverCpotri_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo)
    end function

    function hipsolverCpotri_rank_1(handle,uplo,n,A,lda,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotri_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCpotri_rank_1 = hipsolverCpotri_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo)
    end function

    function hipsolverZpotri_full_rank(handle,uplo,n,A,lda,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotri_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZpotri_full_rank = hipsolverZpotri_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo)
    end function

    function hipsolverZpotri_rank_0(handle,uplo,n,A,lda,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotri_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZpotri_rank_0 = hipsolverZpotri_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo)
    end function

    function hipsolverZpotri_rank_1(handle,uplo,n,A,lda,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotri_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZpotri_rank_1 = hipsolverZpotri_(handle,uplo,n,c_loc(A),lda,work,lwork,devInfo)
    end function

    function hipsolverSpotrs_bufferSize_full_rank(handle,uplo,n,nrhs,A,lda,B,ldb,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrs_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      !
      hipsolverSpotrs_bufferSize_full_rank = hipsolverSpotrs_bufferSize_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,lwork)
    end function

    function hipsolverSpotrs_bufferSize_rank_0(handle,uplo,n,nrhs,A,lda,B,ldb,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrs_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_float),target :: A
      integer(c_int) :: lda
      real(c_float),target :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      !
      hipsolverSpotrs_bufferSize_rank_0 = hipsolverSpotrs_bufferSize_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,lwork)
    end function

    function hipsolverSpotrs_bufferSize_rank_1(handle,uplo,n,nrhs,A,lda,B,ldb,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrs_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:) :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      !
      hipsolverSpotrs_bufferSize_rank_1 = hipsolverSpotrs_bufferSize_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,lwork)
    end function

    function hipsolverDpotrs_bufferSize_full_rank(handle,uplo,n,nrhs,A,lda,B,ldb,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrs_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      !
      hipsolverDpotrs_bufferSize_full_rank = hipsolverDpotrs_bufferSize_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,lwork)
    end function

    function hipsolverDpotrs_bufferSize_rank_0(handle,uplo,n,nrhs,A,lda,B,ldb,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrs_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_double),target :: A
      integer(c_int) :: lda
      real(c_double),target :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      !
      hipsolverDpotrs_bufferSize_rank_0 = hipsolverDpotrs_bufferSize_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,lwork)
    end function

    function hipsolverDpotrs_bufferSize_rank_1(handle,uplo,n,nrhs,A,lda,B,ldb,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrs_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:) :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      !
      hipsolverDpotrs_bufferSize_rank_1 = hipsolverDpotrs_bufferSize_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,lwork)
    end function

    function hipsolverCpotrs_bufferSize_full_rank(handle,uplo,n,nrhs,A,lda,B,ldb,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrs_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_float_complex),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      !
      hipsolverCpotrs_bufferSize_full_rank = hipsolverCpotrs_bufferSize_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,lwork)
    end function

    function hipsolverCpotrs_bufferSize_rank_0(handle,uplo,n,nrhs,A,lda,B,ldb,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrs_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      complex(c_float_complex),target :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      !
      hipsolverCpotrs_bufferSize_rank_0 = hipsolverCpotrs_bufferSize_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,lwork)
    end function

    function hipsolverCpotrs_bufferSize_rank_1(handle,uplo,n,nrhs,A,lda,B,ldb,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrs_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_float_complex),target,dimension(:) :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      !
      hipsolverCpotrs_bufferSize_rank_1 = hipsolverCpotrs_bufferSize_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,lwork)
    end function

    function hipsolverZpotrs_bufferSize_full_rank(handle,uplo,n,nrhs,A,lda,B,ldb,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrs_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_double_complex),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      !
      hipsolverZpotrs_bufferSize_full_rank = hipsolverZpotrs_bufferSize_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,lwork)
    end function

    function hipsolverZpotrs_bufferSize_rank_0(handle,uplo,n,nrhs,A,lda,B,ldb,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrs_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      complex(c_double_complex),target :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      !
      hipsolverZpotrs_bufferSize_rank_0 = hipsolverZpotrs_bufferSize_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,lwork)
    end function

    function hipsolverZpotrs_bufferSize_rank_1(handle,uplo,n,nrhs,A,lda,B,ldb,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrs_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_double_complex),target,dimension(:) :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      !
      hipsolverZpotrs_bufferSize_rank_1 = hipsolverZpotrs_bufferSize_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,lwork)
    end function

    function hipsolverSpotrs_full_rank(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrs_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSpotrs_full_rank = hipsolverSpotrs_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,work,lwork,devInfo)
    end function

    function hipsolverSpotrs_rank_0(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrs_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_float),target :: A
      integer(c_int) :: lda
      real(c_float),target :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSpotrs_rank_0 = hipsolverSpotrs_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,work,lwork,devInfo)
    end function

    function hipsolverSpotrs_rank_1(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrs_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:) :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSpotrs_rank_1 = hipsolverSpotrs_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,work,lwork,devInfo)
    end function

    function hipsolverDpotrs_full_rank(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrs_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDpotrs_full_rank = hipsolverDpotrs_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,work,lwork,devInfo)
    end function

    function hipsolverDpotrs_rank_0(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrs_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_double),target :: A
      integer(c_int) :: lda
      real(c_double),target :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDpotrs_rank_0 = hipsolverDpotrs_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,work,lwork,devInfo)
    end function

    function hipsolverDpotrs_rank_1(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrs_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:) :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDpotrs_rank_1 = hipsolverDpotrs_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,work,lwork,devInfo)
    end function

    function hipsolverCpotrs_full_rank(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrs_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_float_complex),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCpotrs_full_rank = hipsolverCpotrs_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,work,lwork,devInfo)
    end function

    function hipsolverCpotrs_rank_0(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrs_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      complex(c_float_complex),target :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCpotrs_rank_0 = hipsolverCpotrs_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,work,lwork,devInfo)
    end function

    function hipsolverCpotrs_rank_1(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrs_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_float_complex),target,dimension(:) :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCpotrs_rank_1 = hipsolverCpotrs_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,work,lwork,devInfo)
    end function

    function hipsolverZpotrs_full_rank(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrs_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_double_complex),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZpotrs_full_rank = hipsolverZpotrs_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,work,lwork,devInfo)
    end function

    function hipsolverZpotrs_rank_0(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrs_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      complex(c_double_complex),target :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZpotrs_rank_0 = hipsolverZpotrs_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,work,lwork,devInfo)
    end function

    function hipsolverZpotrs_rank_1(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrs_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_double_complex),target,dimension(:) :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZpotrs_rank_1 = hipsolverZpotrs_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,work,lwork,devInfo)
    end function

    function hipsolverSpotrsBatched_bufferSize_full_rank(handle,uplo,n,nrhs,A,lda,B,ldb,lwork,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrsBatched_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_float),target,dimension(:,:,:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:,:,:) :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      integer(c_int) :: batch_count
      !
      hipsolverSpotrsBatched_bufferSize_full_rank = hipsolverSpotrsBatched_bufferSize_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,lwork,batch_count)
    end function

    function hipsolverSpotrsBatched_bufferSize_rank_0(handle,uplo,n,nrhs,A,lda,B,ldb,lwork,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrsBatched_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_float),target :: A
      integer(c_int) :: lda
      real(c_float),target :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      integer(c_int) :: batch_count
      !
      hipsolverSpotrsBatched_bufferSize_rank_0 = hipsolverSpotrsBatched_bufferSize_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,lwork,batch_count)
    end function

    function hipsolverSpotrsBatched_bufferSize_rank_1(handle,uplo,n,nrhs,A,lda,B,ldb,lwork,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrsBatched_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:) :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      integer(c_int) :: batch_count
      !
      hipsolverSpotrsBatched_bufferSize_rank_1 = hipsolverSpotrsBatched_bufferSize_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,lwork,batch_count)
    end function

    function hipsolverDpotrsBatched_bufferSize_full_rank(handle,uplo,n,nrhs,A,lda,B,ldb,lwork,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrsBatched_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_double),target,dimension(:,:,:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:,:,:) :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      integer(c_int) :: batch_count
      !
      hipsolverDpotrsBatched_bufferSize_full_rank = hipsolverDpotrsBatched_bufferSize_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,lwork,batch_count)
    end function

    function hipsolverDpotrsBatched_bufferSize_rank_0(handle,uplo,n,nrhs,A,lda,B,ldb,lwork,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrsBatched_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_double),target :: A
      integer(c_int) :: lda
      real(c_double),target :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      integer(c_int) :: batch_count
      !
      hipsolverDpotrsBatched_bufferSize_rank_0 = hipsolverDpotrsBatched_bufferSize_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,lwork,batch_count)
    end function

    function hipsolverDpotrsBatched_bufferSize_rank_1(handle,uplo,n,nrhs,A,lda,B,ldb,lwork,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrsBatched_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:) :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      integer(c_int) :: batch_count
      !
      hipsolverDpotrsBatched_bufferSize_rank_1 = hipsolverDpotrsBatched_bufferSize_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,lwork,batch_count)
    end function

    function hipsolverCpotrsBatched_bufferSize_full_rank(handle,uplo,n,nrhs,A,lda,B,ldb,lwork,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrsBatched_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_float_complex),target,dimension(:,:,:) :: A
      integer(c_int) :: lda
      complex(c_float_complex),target,dimension(:,:,:) :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      integer(c_int) :: batch_count
      !
      hipsolverCpotrsBatched_bufferSize_full_rank = hipsolverCpotrsBatched_bufferSize_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,lwork,batch_count)
    end function

    function hipsolverCpotrsBatched_bufferSize_rank_0(handle,uplo,n,nrhs,A,lda,B,ldb,lwork,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrsBatched_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      complex(c_float_complex),target :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      integer(c_int) :: batch_count
      !
      hipsolverCpotrsBatched_bufferSize_rank_0 = hipsolverCpotrsBatched_bufferSize_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,lwork,batch_count)
    end function

    function hipsolverCpotrsBatched_bufferSize_rank_1(handle,uplo,n,nrhs,A,lda,B,ldb,lwork,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrsBatched_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_float_complex),target,dimension(:) :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      integer(c_int) :: batch_count
      !
      hipsolverCpotrsBatched_bufferSize_rank_1 = hipsolverCpotrsBatched_bufferSize_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,lwork,batch_count)
    end function

    function hipsolverZpotrsBatched_bufferSize_full_rank(handle,uplo,n,nrhs,A,lda,B,ldb,lwork,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrsBatched_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_double_complex),target,dimension(:,:,:) :: A
      integer(c_int) :: lda
      complex(c_double_complex),target,dimension(:,:,:) :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      integer(c_int) :: batch_count
      !
      hipsolverZpotrsBatched_bufferSize_full_rank = hipsolverZpotrsBatched_bufferSize_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,lwork,batch_count)
    end function

    function hipsolverZpotrsBatched_bufferSize_rank_0(handle,uplo,n,nrhs,A,lda,B,ldb,lwork,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrsBatched_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      complex(c_double_complex),target :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      integer(c_int) :: batch_count
      !
      hipsolverZpotrsBatched_bufferSize_rank_0 = hipsolverZpotrsBatched_bufferSize_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,lwork,batch_count)
    end function

    function hipsolverZpotrsBatched_bufferSize_rank_1(handle,uplo,n,nrhs,A,lda,B,ldb,lwork,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrsBatched_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_double_complex),target,dimension(:) :: B
      integer(c_int) :: ldb
      integer(c_int) :: lwork
      integer(c_int) :: batch_count
      !
      hipsolverZpotrsBatched_bufferSize_rank_1 = hipsolverZpotrsBatched_bufferSize_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,lwork,batch_count)
    end function

    function hipsolverSpotrsBatched_full_rank(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrsBatched_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_float),target,dimension(:,:,:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:,:,:) :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      integer(c_int) :: batch_count
      !
      hipsolverSpotrsBatched_full_rank = hipsolverSpotrsBatched_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,work,lwork,devInfo,batch_count)
    end function

    function hipsolverSpotrsBatched_rank_0(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrsBatched_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_float),target :: A
      integer(c_int) :: lda
      real(c_float),target :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      integer(c_int) :: batch_count
      !
      hipsolverSpotrsBatched_rank_0 = hipsolverSpotrsBatched_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,work,lwork,devInfo,batch_count)
    end function

    function hipsolverSpotrsBatched_rank_1(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrsBatched_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:) :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      integer(c_int) :: batch_count
      !
      hipsolverSpotrsBatched_rank_1 = hipsolverSpotrsBatched_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,work,lwork,devInfo,batch_count)
    end function

    function hipsolverDpotrsBatched_full_rank(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrsBatched_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_double),target,dimension(:,:,:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:,:,:) :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      integer(c_int) :: batch_count
      !
      hipsolverDpotrsBatched_full_rank = hipsolverDpotrsBatched_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,work,lwork,devInfo,batch_count)
    end function

    function hipsolverDpotrsBatched_rank_0(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrsBatched_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_double),target :: A
      integer(c_int) :: lda
      real(c_double),target :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      integer(c_int) :: batch_count
      !
      hipsolverDpotrsBatched_rank_0 = hipsolverDpotrsBatched_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,work,lwork,devInfo,batch_count)
    end function

    function hipsolverDpotrsBatched_rank_1(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrsBatched_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:) :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      integer(c_int) :: batch_count
      !
      hipsolverDpotrsBatched_rank_1 = hipsolverDpotrsBatched_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,work,lwork,devInfo,batch_count)
    end function

    function hipsolverCpotrsBatched_full_rank(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrsBatched_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_float_complex),target,dimension(:,:,:) :: A
      integer(c_int) :: lda
      complex(c_float_complex),target,dimension(:,:,:) :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      integer(c_int) :: batch_count
      !
      hipsolverCpotrsBatched_full_rank = hipsolverCpotrsBatched_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,work,lwork,devInfo,batch_count)
    end function

    function hipsolverCpotrsBatched_rank_0(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrsBatched_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      complex(c_float_complex),target :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      integer(c_int) :: batch_count
      !
      hipsolverCpotrsBatched_rank_0 = hipsolverCpotrsBatched_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,work,lwork,devInfo,batch_count)
    end function

    function hipsolverCpotrsBatched_rank_1(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrsBatched_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_float_complex),target,dimension(:) :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      integer(c_int) :: batch_count
      !
      hipsolverCpotrsBatched_rank_1 = hipsolverCpotrsBatched_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,work,lwork,devInfo,batch_count)
    end function

    function hipsolverZpotrsBatched_full_rank(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrsBatched_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_double_complex),target,dimension(:,:,:) :: A
      integer(c_int) :: lda
      complex(c_double_complex),target,dimension(:,:,:) :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      integer(c_int) :: batch_count
      !
      hipsolverZpotrsBatched_full_rank = hipsolverZpotrsBatched_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,work,lwork,devInfo,batch_count)
    end function

    function hipsolverZpotrsBatched_rank_0(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrsBatched_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      complex(c_double_complex),target :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      integer(c_int) :: batch_count
      !
      hipsolverZpotrsBatched_rank_0 = hipsolverZpotrsBatched_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,work,lwork,devInfo,batch_count)
    end function

    function hipsolverZpotrsBatched_rank_1(handle,uplo,n,nrhs,A,lda,B,ldb,work,lwork,devInfo,batch_count)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrsBatched_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      integer(c_int) :: nrhs
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_double_complex),target,dimension(:) :: B
      integer(c_int) :: ldb
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      integer(c_int) :: batch_count
      !
      hipsolverZpotrsBatched_rank_1 = hipsolverZpotrsBatched_(handle,uplo,n,nrhs,c_loc(A),lda,c_loc(B),ldb,work,lwork,devInfo,batch_count)
    end function

    function hipsolverSsyevd_bufferSize_full_rank(handle,jobz,uplo,n,A,lda,D,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsyevd_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:) :: D
      integer(c_int) :: lwork
      !
      hipsolverSsyevd_bufferSize_full_rank = hipsolverSsyevd_bufferSize_(handle,jobz,uplo,n,c_loc(A),lda,c_loc(D),lwork)
    end function

    function hipsolverSsyevd_bufferSize_rank_0(handle,jobz,uplo,n,A,lda,D,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsyevd_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target :: A
      integer(c_int) :: lda
      real(c_float),target :: D
      integer(c_int) :: lwork
      !
      hipsolverSsyevd_bufferSize_rank_0 = hipsolverSsyevd_bufferSize_(handle,jobz,uplo,n,c_loc(A),lda,c_loc(D),lwork)
    end function

    function hipsolverSsyevd_bufferSize_rank_1(handle,jobz,uplo,n,A,lda,D,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsyevd_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:) :: D
      integer(c_int) :: lwork
      !
      hipsolverSsyevd_bufferSize_rank_1 = hipsolverSsyevd_bufferSize_(handle,jobz,uplo,n,c_loc(A),lda,c_loc(D),lwork)
    end function

    function hipsolverDsyevd_bufferSize_full_rank(handle,jobz,uplo,n,A,lda,D,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsyevd_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:) :: D
      integer(c_int) :: lwork
      !
      hipsolverDsyevd_bufferSize_full_rank = hipsolverDsyevd_bufferSize_(handle,jobz,uplo,n,c_loc(A),lda,c_loc(D),lwork)
    end function

    function hipsolverDsyevd_bufferSize_rank_0(handle,jobz,uplo,n,A,lda,D,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsyevd_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target :: A
      integer(c_int) :: lda
      real(c_double),target :: D
      integer(c_int) :: lwork
      !
      hipsolverDsyevd_bufferSize_rank_0 = hipsolverDsyevd_bufferSize_(handle,jobz,uplo,n,c_loc(A),lda,c_loc(D),lwork)
    end function

    function hipsolverDsyevd_bufferSize_rank_1(handle,jobz,uplo,n,A,lda,D,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsyevd_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:) :: D
      integer(c_int) :: lwork
      !
      hipsolverDsyevd_bufferSize_rank_1 = hipsolverDsyevd_bufferSize_(handle,jobz,uplo,n,c_loc(A),lda,c_loc(D),lwork)
    end function

    function hipsolverCheevd_bufferSize_full_rank(handle,jobz,uplo,n,A,lda,D,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCheevd_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:) :: D
      integer(c_int) :: lwork
      !
      hipsolverCheevd_bufferSize_full_rank = hipsolverCheevd_bufferSize_(handle,jobz,uplo,n,c_loc(A),lda,c_loc(D),lwork)
    end function

    function hipsolverCheevd_bufferSize_rank_0(handle,jobz,uplo,n,A,lda,D,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCheevd_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      real(c_float),target :: D
      integer(c_int) :: lwork
      !
      hipsolverCheevd_bufferSize_rank_0 = hipsolverCheevd_bufferSize_(handle,jobz,uplo,n,c_loc(A),lda,c_loc(D),lwork)
    end function

    function hipsolverCheevd_bufferSize_rank_1(handle,jobz,uplo,n,A,lda,D,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCheevd_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:) :: D
      integer(c_int) :: lwork
      !
      hipsolverCheevd_bufferSize_rank_1 = hipsolverCheevd_bufferSize_(handle,jobz,uplo,n,c_loc(A),lda,c_loc(D),lwork)
    end function

    function hipsolverZheevd_bufferSize_full_rank(handle,jobz,uplo,n,A,lda,D,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZheevd_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:) :: D
      integer(c_int) :: lwork
      !
      hipsolverZheevd_bufferSize_full_rank = hipsolverZheevd_bufferSize_(handle,jobz,uplo,n,c_loc(A),lda,c_loc(D),lwork)
    end function

    function hipsolverZheevd_bufferSize_rank_0(handle,jobz,uplo,n,A,lda,D,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZheevd_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      real(c_double),target :: D
      integer(c_int) :: lwork
      !
      hipsolverZheevd_bufferSize_rank_0 = hipsolverZheevd_bufferSize_(handle,jobz,uplo,n,c_loc(A),lda,c_loc(D),lwork)
    end function

    function hipsolverZheevd_bufferSize_rank_1(handle,jobz,uplo,n,A,lda,D,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZheevd_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:) :: D
      integer(c_int) :: lwork
      !
      hipsolverZheevd_bufferSize_rank_1 = hipsolverZheevd_bufferSize_(handle,jobz,uplo,n,c_loc(A),lda,c_loc(D),lwork)
    end function

    function hipsolverSsyevd_full_rank(handle,jobz,uplo,n,A,lda,D,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsyevd_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:) :: D
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSsyevd_full_rank = hipsolverSsyevd_(handle,jobz,uplo,n,c_loc(A),lda,c_loc(D),work,lwork,devInfo)
    end function

    function hipsolverSsyevd_rank_0(handle,jobz,uplo,n,A,lda,D,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsyevd_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target :: A
      integer(c_int) :: lda
      real(c_float),target :: D
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSsyevd_rank_0 = hipsolverSsyevd_(handle,jobz,uplo,n,c_loc(A),lda,c_loc(D),work,lwork,devInfo)
    end function

    function hipsolverSsyevd_rank_1(handle,jobz,uplo,n,A,lda,D,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsyevd_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:) :: D
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSsyevd_rank_1 = hipsolverSsyevd_(handle,jobz,uplo,n,c_loc(A),lda,c_loc(D),work,lwork,devInfo)
    end function

    function hipsolverDsyevd_full_rank(handle,jobz,uplo,n,A,lda,D,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsyevd_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:) :: D
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDsyevd_full_rank = hipsolverDsyevd_(handle,jobz,uplo,n,c_loc(A),lda,c_loc(D),work,lwork,devInfo)
    end function

    function hipsolverDsyevd_rank_0(handle,jobz,uplo,n,A,lda,D,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsyevd_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target :: A
      integer(c_int) :: lda
      real(c_double),target :: D
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDsyevd_rank_0 = hipsolverDsyevd_(handle,jobz,uplo,n,c_loc(A),lda,c_loc(D),work,lwork,devInfo)
    end function

    function hipsolverDsyevd_rank_1(handle,jobz,uplo,n,A,lda,D,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsyevd_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:) :: D
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDsyevd_rank_1 = hipsolverDsyevd_(handle,jobz,uplo,n,c_loc(A),lda,c_loc(D),work,lwork,devInfo)
    end function

    function hipsolverCheevd_full_rank(handle,jobz,uplo,n,A,lda,D,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCheevd_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:) :: D
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCheevd_full_rank = hipsolverCheevd_(handle,jobz,uplo,n,c_loc(A),lda,c_loc(D),work,lwork,devInfo)
    end function

    function hipsolverCheevd_rank_0(handle,jobz,uplo,n,A,lda,D,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCheevd_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      real(c_float),target :: D
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCheevd_rank_0 = hipsolverCheevd_(handle,jobz,uplo,n,c_loc(A),lda,c_loc(D),work,lwork,devInfo)
    end function

    function hipsolverCheevd_rank_1(handle,jobz,uplo,n,A,lda,D,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCheevd_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:) :: D
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCheevd_rank_1 = hipsolverCheevd_(handle,jobz,uplo,n,c_loc(A),lda,c_loc(D),work,lwork,devInfo)
    end function

    function hipsolverZheevd_full_rank(handle,jobz,uplo,n,A,lda,D,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZheevd_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:) :: D
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZheevd_full_rank = hipsolverZheevd_(handle,jobz,uplo,n,c_loc(A),lda,c_loc(D),work,lwork,devInfo)
    end function

    function hipsolverZheevd_rank_0(handle,jobz,uplo,n,A,lda,D,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZheevd_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      real(c_double),target :: D
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZheevd_rank_0 = hipsolverZheevd_(handle,jobz,uplo,n,c_loc(A),lda,c_loc(D),work,lwork,devInfo)
    end function

    function hipsolverZheevd_rank_1(handle,jobz,uplo,n,A,lda,D,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZheevd_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:) :: D
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZheevd_rank_1 = hipsolverZheevd_(handle,jobz,uplo,n,c_loc(A),lda,c_loc(D),work,lwork,devInfo)
    end function

    function hipsolverSsygvd_bufferSize_full_rank(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsygvd_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)) :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      real(c_float),target,dimension(:) :: D
      integer(c_int) :: lwork
      !
      hipsolverSsygvd_bufferSize_full_rank = hipsolverSsygvd_bufferSize_(handle,itype,jobz,uplo,n,c_loc(A),lda,c_loc(B),ldb,c_loc(D),lwork)
    end function

    function hipsolverSsygvd_bufferSize_rank_0(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsygvd_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)) :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target :: A
      integer(c_int) :: lda
      real(c_float),target :: B
      integer(c_int) :: ldb
      real(c_float),target :: D
      integer(c_int) :: lwork
      !
      hipsolverSsygvd_bufferSize_rank_0 = hipsolverSsygvd_bufferSize_(handle,itype,jobz,uplo,n,c_loc(A),lda,c_loc(B),ldb,c_loc(D),lwork)
    end function

    function hipsolverSsygvd_bufferSize_rank_1(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsygvd_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)) :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:) :: B
      integer(c_int) :: ldb
      real(c_float),target,dimension(:) :: D
      integer(c_int) :: lwork
      !
      hipsolverSsygvd_bufferSize_rank_1 = hipsolverSsygvd_bufferSize_(handle,itype,jobz,uplo,n,c_loc(A),lda,c_loc(B),ldb,c_loc(D),lwork)
    end function

    function hipsolverDsygvd_bufferSize_full_rank(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsygvd_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)) :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      real(c_double),target,dimension(:) :: D
      integer(c_int) :: lwork
      !
      hipsolverDsygvd_bufferSize_full_rank = hipsolverDsygvd_bufferSize_(handle,itype,jobz,uplo,n,c_loc(A),lda,c_loc(B),ldb,c_loc(D),lwork)
    end function

    function hipsolverDsygvd_bufferSize_rank_0(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsygvd_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)) :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target :: A
      integer(c_int) :: lda
      real(c_double),target :: B
      integer(c_int) :: ldb
      real(c_double),target :: D
      integer(c_int) :: lwork
      !
      hipsolverDsygvd_bufferSize_rank_0 = hipsolverDsygvd_bufferSize_(handle,itype,jobz,uplo,n,c_loc(A),lda,c_loc(B),ldb,c_loc(D),lwork)
    end function

    function hipsolverDsygvd_bufferSize_rank_1(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsygvd_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)) :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:) :: B
      integer(c_int) :: ldb
      real(c_double),target,dimension(:) :: D
      integer(c_int) :: lwork
      !
      hipsolverDsygvd_bufferSize_rank_1 = hipsolverDsygvd_bufferSize_(handle,itype,jobz,uplo,n,c_loc(A),lda,c_loc(B),ldb,c_loc(D),lwork)
    end function

    function hipsolverChegvd_bufferSize_full_rank(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverChegvd_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)) :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_float_complex),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      real(c_float),target,dimension(:) :: D
      integer(c_int) :: lwork
      !
      hipsolverChegvd_bufferSize_full_rank = hipsolverChegvd_bufferSize_(handle,itype,jobz,uplo,n,c_loc(A),lda,c_loc(B),ldb,c_loc(D),lwork)
    end function

    function hipsolverChegvd_bufferSize_rank_0(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverChegvd_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)) :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      complex(c_float_complex),target :: B
      integer(c_int) :: ldb
      real(c_float),target :: D
      integer(c_int) :: lwork
      !
      hipsolverChegvd_bufferSize_rank_0 = hipsolverChegvd_bufferSize_(handle,itype,jobz,uplo,n,c_loc(A),lda,c_loc(B),ldb,c_loc(D),lwork)
    end function

    function hipsolverChegvd_bufferSize_rank_1(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverChegvd_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)) :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_float_complex),target,dimension(:) :: B
      integer(c_int) :: ldb
      real(c_float),target,dimension(:) :: D
      integer(c_int) :: lwork
      !
      hipsolverChegvd_bufferSize_rank_1 = hipsolverChegvd_bufferSize_(handle,itype,jobz,uplo,n,c_loc(A),lda,c_loc(B),ldb,c_loc(D),lwork)
    end function

    function hipsolverZhegvd_bufferSize_full_rank(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZhegvd_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)) :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_double_complex),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      real(c_double),target,dimension(:) :: D
      integer(c_int) :: lwork
      !
      hipsolverZhegvd_bufferSize_full_rank = hipsolverZhegvd_bufferSize_(handle,itype,jobz,uplo,n,c_loc(A),lda,c_loc(B),ldb,c_loc(D),lwork)
    end function

    function hipsolverZhegvd_bufferSize_rank_0(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZhegvd_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)) :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      complex(c_double_complex),target :: B
      integer(c_int) :: ldb
      real(c_double),target :: D
      integer(c_int) :: lwork
      !
      hipsolverZhegvd_bufferSize_rank_0 = hipsolverZhegvd_bufferSize_(handle,itype,jobz,uplo,n,c_loc(A),lda,c_loc(B),ldb,c_loc(D),lwork)
    end function

    function hipsolverZhegvd_bufferSize_rank_1(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZhegvd_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)) :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_double_complex),target,dimension(:) :: B
      integer(c_int) :: ldb
      real(c_double),target,dimension(:) :: D
      integer(c_int) :: lwork
      !
      hipsolverZhegvd_bufferSize_rank_1 = hipsolverZhegvd_bufferSize_(handle,itype,jobz,uplo,n,c_loc(A),lda,c_loc(B),ldb,c_loc(D),lwork)
    end function

    function hipsolverSsygvd_full_rank(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsygvd_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)) :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      real(c_float),target,dimension(:) :: D
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSsygvd_full_rank = hipsolverSsygvd_(handle,itype,jobz,uplo,n,c_loc(A),lda,c_loc(B),ldb,c_loc(D),work,lwork,devInfo)
    end function

    function hipsolverSsygvd_rank_0(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsygvd_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)) :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target :: A
      integer(c_int) :: lda
      real(c_float),target :: B
      integer(c_int) :: ldb
      real(c_float),target :: D
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSsygvd_rank_0 = hipsolverSsygvd_(handle,itype,jobz,uplo,n,c_loc(A),lda,c_loc(B),ldb,c_loc(D),work,lwork,devInfo)
    end function

    function hipsolverSsygvd_rank_1(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsygvd_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)) :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:) :: B
      integer(c_int) :: ldb
      real(c_float),target,dimension(:) :: D
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSsygvd_rank_1 = hipsolverSsygvd_(handle,itype,jobz,uplo,n,c_loc(A),lda,c_loc(B),ldb,c_loc(D),work,lwork,devInfo)
    end function

    function hipsolverDsygvd_full_rank(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsygvd_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)) :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      real(c_double),target,dimension(:) :: D
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDsygvd_full_rank = hipsolverDsygvd_(handle,itype,jobz,uplo,n,c_loc(A),lda,c_loc(B),ldb,c_loc(D),work,lwork,devInfo)
    end function

    function hipsolverDsygvd_rank_0(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsygvd_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)) :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target :: A
      integer(c_int) :: lda
      real(c_double),target :: B
      integer(c_int) :: ldb
      real(c_double),target :: D
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDsygvd_rank_0 = hipsolverDsygvd_(handle,itype,jobz,uplo,n,c_loc(A),lda,c_loc(B),ldb,c_loc(D),work,lwork,devInfo)
    end function

    function hipsolverDsygvd_rank_1(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsygvd_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)) :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:) :: B
      integer(c_int) :: ldb
      real(c_double),target,dimension(:) :: D
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDsygvd_rank_1 = hipsolverDsygvd_(handle,itype,jobz,uplo,n,c_loc(A),lda,c_loc(B),ldb,c_loc(D),work,lwork,devInfo)
    end function

    function hipsolverChegvd_full_rank(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverChegvd_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)) :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_float_complex),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      real(c_float),target,dimension(:) :: D
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverChegvd_full_rank = hipsolverChegvd_(handle,itype,jobz,uplo,n,c_loc(A),lda,c_loc(B),ldb,c_loc(D),work,lwork,devInfo)
    end function

    function hipsolverChegvd_rank_0(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverChegvd_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)) :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      complex(c_float_complex),target :: B
      integer(c_int) :: ldb
      real(c_float),target :: D
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverChegvd_rank_0 = hipsolverChegvd_(handle,itype,jobz,uplo,n,c_loc(A),lda,c_loc(B),ldb,c_loc(D),work,lwork,devInfo)
    end function

    function hipsolverChegvd_rank_1(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverChegvd_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)) :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_float_complex),target,dimension(:) :: B
      integer(c_int) :: ldb
      real(c_float),target,dimension(:) :: D
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverChegvd_rank_1 = hipsolverChegvd_(handle,itype,jobz,uplo,n,c_loc(A),lda,c_loc(B),ldb,c_loc(D),work,lwork,devInfo)
    end function

    function hipsolverZhegvd_full_rank(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZhegvd_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)) :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      complex(c_double_complex),target,dimension(:,:) :: B
      integer(c_int) :: ldb
      real(c_double),target,dimension(:) :: D
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZhegvd_full_rank = hipsolverZhegvd_(handle,itype,jobz,uplo,n,c_loc(A),lda,c_loc(B),ldb,c_loc(D),work,lwork,devInfo)
    end function

    function hipsolverZhegvd_rank_0(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZhegvd_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)) :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      complex(c_double_complex),target :: B
      integer(c_int) :: ldb
      real(c_double),target :: D
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZhegvd_rank_0 = hipsolverZhegvd_(handle,itype,jobz,uplo,n,c_loc(A),lda,c_loc(B),ldb,c_loc(D),work,lwork,devInfo)
    end function

    function hipsolverZhegvd_rank_1(handle,itype,jobz,uplo,n,A,lda,B,ldb,D,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZhegvd_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_EIG_TYPE_1)) :: itype
      integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)) :: jobz
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      complex(c_double_complex),target,dimension(:) :: B
      integer(c_int) :: ldb
      real(c_double),target,dimension(:) :: D
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZhegvd_rank_1 = hipsolverZhegvd_(handle,itype,jobz,uplo,n,c_loc(A),lda,c_loc(B),ldb,c_loc(D),work,lwork,devInfo)
    end function

    function hipsolverSsytrd_bufferSize_full_rank(handle,uplo,n,A,lda,D,E,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsytrd_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:) :: D
      real(c_float),target,dimension(:) :: E
      real(c_float) :: tau
      integer(c_int) :: lwork
      !
      hipsolverSsytrd_bufferSize_full_rank = hipsolverSsytrd_bufferSize_(handle,uplo,n,c_loc(A),lda,c_loc(D),c_loc(E),tau,lwork)
    end function

    function hipsolverSsytrd_bufferSize_rank_0(handle,uplo,n,A,lda,D,E,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsytrd_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target :: A
      integer(c_int) :: lda
      real(c_float),target :: D
      real(c_float),target :: E
      real(c_float) :: tau
      integer(c_int) :: lwork
      !
      hipsolverSsytrd_bufferSize_rank_0 = hipsolverSsytrd_bufferSize_(handle,uplo,n,c_loc(A),lda,c_loc(D),c_loc(E),tau,lwork)
    end function

    function hipsolverSsytrd_bufferSize_rank_1(handle,uplo,n,A,lda,D,E,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsytrd_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:) :: D
      real(c_float),target,dimension(:) :: E
      real(c_float) :: tau
      integer(c_int) :: lwork
      !
      hipsolverSsytrd_bufferSize_rank_1 = hipsolverSsytrd_bufferSize_(handle,uplo,n,c_loc(A),lda,c_loc(D),c_loc(E),tau,lwork)
    end function

    function hipsolverDsytrd_bufferSize_full_rank(handle,uplo,n,A,lda,D,E,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsytrd_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:) :: D
      real(c_double),target,dimension(:) :: E
      real(c_double) :: tau
      integer(c_int) :: lwork
      !
      hipsolverDsytrd_bufferSize_full_rank = hipsolverDsytrd_bufferSize_(handle,uplo,n,c_loc(A),lda,c_loc(D),c_loc(E),tau,lwork)
    end function

    function hipsolverDsytrd_bufferSize_rank_0(handle,uplo,n,A,lda,D,E,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsytrd_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target :: A
      integer(c_int) :: lda
      real(c_double),target :: D
      real(c_double),target :: E
      real(c_double) :: tau
      integer(c_int) :: lwork
      !
      hipsolverDsytrd_bufferSize_rank_0 = hipsolverDsytrd_bufferSize_(handle,uplo,n,c_loc(A),lda,c_loc(D),c_loc(E),tau,lwork)
    end function

    function hipsolverDsytrd_bufferSize_rank_1(handle,uplo,n,A,lda,D,E,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsytrd_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:) :: D
      real(c_double),target,dimension(:) :: E
      real(c_double) :: tau
      integer(c_int) :: lwork
      !
      hipsolverDsytrd_bufferSize_rank_1 = hipsolverDsytrd_bufferSize_(handle,uplo,n,c_loc(A),lda,c_loc(D),c_loc(E),tau,lwork)
    end function

    function hipsolverChetrd_bufferSize_full_rank(handle,uplo,n,A,lda,D,E,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverChetrd_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:) :: D
      real(c_float),target,dimension(:) :: E
      complex(c_float_complex) :: tau
      integer(c_int) :: lwork
      !
      hipsolverChetrd_bufferSize_full_rank = hipsolverChetrd_bufferSize_(handle,uplo,n,c_loc(A),lda,c_loc(D),c_loc(E),tau,lwork)
    end function

    function hipsolverChetrd_bufferSize_rank_0(handle,uplo,n,A,lda,D,E,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverChetrd_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      real(c_float),target :: D
      real(c_float),target :: E
      complex(c_float_complex) :: tau
      integer(c_int) :: lwork
      !
      hipsolverChetrd_bufferSize_rank_0 = hipsolverChetrd_bufferSize_(handle,uplo,n,c_loc(A),lda,c_loc(D),c_loc(E),tau,lwork)
    end function

    function hipsolverChetrd_bufferSize_rank_1(handle,uplo,n,A,lda,D,E,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverChetrd_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:) :: D
      real(c_float),target,dimension(:) :: E
      complex(c_float_complex) :: tau
      integer(c_int) :: lwork
      !
      hipsolverChetrd_bufferSize_rank_1 = hipsolverChetrd_bufferSize_(handle,uplo,n,c_loc(A),lda,c_loc(D),c_loc(E),tau,lwork)
    end function

    function hipsolverZhetrd_bufferSize_full_rank(handle,uplo,n,A,lda,D,E,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZhetrd_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:) :: D
      real(c_double),target,dimension(:) :: E
      complex(c_double_complex) :: tau
      integer(c_int) :: lwork
      !
      hipsolverZhetrd_bufferSize_full_rank = hipsolverZhetrd_bufferSize_(handle,uplo,n,c_loc(A),lda,c_loc(D),c_loc(E),tau,lwork)
    end function

    function hipsolverZhetrd_bufferSize_rank_0(handle,uplo,n,A,lda,D,E,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZhetrd_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      real(c_double),target :: D
      real(c_double),target :: E
      complex(c_double_complex) :: tau
      integer(c_int) :: lwork
      !
      hipsolverZhetrd_bufferSize_rank_0 = hipsolverZhetrd_bufferSize_(handle,uplo,n,c_loc(A),lda,c_loc(D),c_loc(E),tau,lwork)
    end function

    function hipsolverZhetrd_bufferSize_rank_1(handle,uplo,n,A,lda,D,E,tau,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZhetrd_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:) :: D
      real(c_double),target,dimension(:) :: E
      complex(c_double_complex) :: tau
      integer(c_int) :: lwork
      !
      hipsolverZhetrd_bufferSize_rank_1 = hipsolverZhetrd_bufferSize_(handle,uplo,n,c_loc(A),lda,c_loc(D),c_loc(E),tau,lwork)
    end function

    function hipsolverSsytrd_full_rank(handle,uplo,n,A,lda,D,E,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsytrd_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:) :: D
      real(c_float),target,dimension(:) :: E
      real(c_float) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSsytrd_full_rank = hipsolverSsytrd_(handle,uplo,n,c_loc(A),lda,c_loc(D),c_loc(E),tau,work,lwork,devInfo)
    end function

    function hipsolverSsytrd_rank_0(handle,uplo,n,A,lda,D,E,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsytrd_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target :: A
      integer(c_int) :: lda
      real(c_float),target :: D
      real(c_float),target :: E
      real(c_float) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSsytrd_rank_0 = hipsolverSsytrd_(handle,uplo,n,c_loc(A),lda,c_loc(D),c_loc(E),tau,work,lwork,devInfo)
    end function

    function hipsolverSsytrd_rank_1(handle,uplo,n,A,lda,D,E,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsytrd_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:) :: D
      real(c_float),target,dimension(:) :: E
      real(c_float) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSsytrd_rank_1 = hipsolverSsytrd_(handle,uplo,n,c_loc(A),lda,c_loc(D),c_loc(E),tau,work,lwork,devInfo)
    end function

    function hipsolverDsytrd_full_rank(handle,uplo,n,A,lda,D,E,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsytrd_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:) :: D
      real(c_double),target,dimension(:) :: E
      real(c_double) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDsytrd_full_rank = hipsolverDsytrd_(handle,uplo,n,c_loc(A),lda,c_loc(D),c_loc(E),tau,work,lwork,devInfo)
    end function

    function hipsolverDsytrd_rank_0(handle,uplo,n,A,lda,D,E,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsytrd_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target :: A
      integer(c_int) :: lda
      real(c_double),target :: D
      real(c_double),target :: E
      real(c_double) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDsytrd_rank_0 = hipsolverDsytrd_(handle,uplo,n,c_loc(A),lda,c_loc(D),c_loc(E),tau,work,lwork,devInfo)
    end function

    function hipsolverDsytrd_rank_1(handle,uplo,n,A,lda,D,E,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsytrd_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:) :: D
      real(c_double),target,dimension(:) :: E
      real(c_double) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDsytrd_rank_1 = hipsolverDsytrd_(handle,uplo,n,c_loc(A),lda,c_loc(D),c_loc(E),tau,work,lwork,devInfo)
    end function

    function hipsolverChetrd_full_rank(handle,uplo,n,A,lda,D,E,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverChetrd_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:) :: D
      real(c_float),target,dimension(:) :: E
      complex(c_float_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverChetrd_full_rank = hipsolverChetrd_(handle,uplo,n,c_loc(A),lda,c_loc(D),c_loc(E),tau,work,lwork,devInfo)
    end function

    function hipsolverChetrd_rank_0(handle,uplo,n,A,lda,D,E,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverChetrd_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      real(c_float),target :: D
      real(c_float),target :: E
      complex(c_float_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverChetrd_rank_0 = hipsolverChetrd_(handle,uplo,n,c_loc(A),lda,c_loc(D),c_loc(E),tau,work,lwork,devInfo)
    end function

    function hipsolverChetrd_rank_1(handle,uplo,n,A,lda,D,E,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverChetrd_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_float),target,dimension(:) :: D
      real(c_float),target,dimension(:) :: E
      complex(c_float_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverChetrd_rank_1 = hipsolverChetrd_(handle,uplo,n,c_loc(A),lda,c_loc(D),c_loc(E),tau,work,lwork,devInfo)
    end function

    function hipsolverZhetrd_full_rank(handle,uplo,n,A,lda,D,E,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZhetrd_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:) :: D
      real(c_double),target,dimension(:) :: E
      complex(c_double_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZhetrd_full_rank = hipsolverZhetrd_(handle,uplo,n,c_loc(A),lda,c_loc(D),c_loc(E),tau,work,lwork,devInfo)
    end function

    function hipsolverZhetrd_rank_0(handle,uplo,n,A,lda,D,E,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZhetrd_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      real(c_double),target :: D
      real(c_double),target :: E
      complex(c_double_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZhetrd_rank_0 = hipsolverZhetrd_(handle,uplo,n,c_loc(A),lda,c_loc(D),c_loc(E),tau,work,lwork,devInfo)
    end function

    function hipsolverZhetrd_rank_1(handle,uplo,n,A,lda,D,E,tau,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZhetrd_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      real(c_double),target,dimension(:) :: D
      real(c_double),target,dimension(:) :: E
      complex(c_double_complex) :: tau
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZhetrd_rank_1 = hipsolverZhetrd_(handle,uplo,n,c_loc(A),lda,c_loc(D),c_loc(E),tau,work,lwork,devInfo)
    end function

    function hipsolverSsytrf_bufferSize_full_rank(handle,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsytrf_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: n
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverSsytrf_bufferSize_full_rank = hipsolverSsytrf_bufferSize_(handle,n,c_loc(A),lda,lwork)
    end function

    function hipsolverSsytrf_bufferSize_rank_0(handle,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsytrf_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: n
      real(c_float),target :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverSsytrf_bufferSize_rank_0 = hipsolverSsytrf_bufferSize_(handle,n,c_loc(A),lda,lwork)
    end function

    function hipsolverSsytrf_bufferSize_rank_1(handle,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsytrf_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: n
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverSsytrf_bufferSize_rank_1 = hipsolverSsytrf_bufferSize_(handle,n,c_loc(A),lda,lwork)
    end function

    function hipsolverDsytrf_bufferSize_full_rank(handle,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsytrf_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: n
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverDsytrf_bufferSize_full_rank = hipsolverDsytrf_bufferSize_(handle,n,c_loc(A),lda,lwork)
    end function

    function hipsolverDsytrf_bufferSize_rank_0(handle,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsytrf_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: n
      real(c_double),target :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverDsytrf_bufferSize_rank_0 = hipsolverDsytrf_bufferSize_(handle,n,c_loc(A),lda,lwork)
    end function

    function hipsolverDsytrf_bufferSize_rank_1(handle,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsytrf_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: n
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverDsytrf_bufferSize_rank_1 = hipsolverDsytrf_bufferSize_(handle,n,c_loc(A),lda,lwork)
    end function

    function hipsolverCsytrf_bufferSize_full_rank(handle,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCsytrf_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverCsytrf_bufferSize_full_rank = hipsolverCsytrf_bufferSize_(handle,n,c_loc(A),lda,lwork)
    end function

    function hipsolverCsytrf_bufferSize_rank_0(handle,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCsytrf_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: n
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverCsytrf_bufferSize_rank_0 = hipsolverCsytrf_bufferSize_(handle,n,c_loc(A),lda,lwork)
    end function

    function hipsolverCsytrf_bufferSize_rank_1(handle,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCsytrf_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverCsytrf_bufferSize_rank_1 = hipsolverCsytrf_bufferSize_(handle,n,c_loc(A),lda,lwork)
    end function

    function hipsolverZsytrf_bufferSize_full_rank(handle,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZsytrf_bufferSize_full_rank
      type(c_ptr) :: handle
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverZsytrf_bufferSize_full_rank = hipsolverZsytrf_bufferSize_(handle,n,c_loc(A),lda,lwork)
    end function

    function hipsolverZsytrf_bufferSize_rank_0(handle,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZsytrf_bufferSize_rank_0
      type(c_ptr) :: handle
      integer(c_int) :: n
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverZsytrf_bufferSize_rank_0 = hipsolverZsytrf_bufferSize_(handle,n,c_loc(A),lda,lwork)
    end function

    function hipsolverZsytrf_bufferSize_rank_1(handle,n,A,lda,lwork)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZsytrf_bufferSize_rank_1
      type(c_ptr) :: handle
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      integer(c_int) :: lwork
      !
      hipsolverZsytrf_bufferSize_rank_1 = hipsolverZsytrf_bufferSize_(handle,n,c_loc(A),lda,lwork)
    end function

    function hipsolverSsytrf_full_rank(handle,uplo,n,A,lda,ipiv,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsytrf_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:,:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: ipiv
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSsytrf_full_rank = hipsolverSsytrf_(handle,uplo,n,c_loc(A),lda,ipiv,work,lwork,devInfo)
    end function

    function hipsolverSsytrf_rank_0(handle,uplo,n,A,lda,ipiv,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsytrf_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target :: A
      integer(c_int) :: lda
      type(c_ptr) :: ipiv
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSsytrf_rank_0 = hipsolverSsytrf_(handle,uplo,n,c_loc(A),lda,ipiv,work,lwork,devInfo)
    end function

    function hipsolverSsytrf_rank_1(handle,uplo,n,A,lda,ipiv,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsytrf_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_float),target,dimension(:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: ipiv
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverSsytrf_rank_1 = hipsolverSsytrf_(handle,uplo,n,c_loc(A),lda,ipiv,work,lwork,devInfo)
    end function

    function hipsolverDsytrf_full_rank(handle,uplo,n,A,lda,ipiv,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsytrf_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:,:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: ipiv
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDsytrf_full_rank = hipsolverDsytrf_(handle,uplo,n,c_loc(A),lda,ipiv,work,lwork,devInfo)
    end function

    function hipsolverDsytrf_rank_0(handle,uplo,n,A,lda,ipiv,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsytrf_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target :: A
      integer(c_int) :: lda
      type(c_ptr) :: ipiv
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDsytrf_rank_0 = hipsolverDsytrf_(handle,uplo,n,c_loc(A),lda,ipiv,work,lwork,devInfo)
    end function

    function hipsolverDsytrf_rank_1(handle,uplo,n,A,lda,ipiv,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsytrf_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      real(c_double),target,dimension(:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: ipiv
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverDsytrf_rank_1 = hipsolverDsytrf_(handle,uplo,n,c_loc(A),lda,ipiv,work,lwork,devInfo)
    end function

    function hipsolverCsytrf_full_rank(handle,uplo,n,A,lda,ipiv,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCsytrf_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: ipiv
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCsytrf_full_rank = hipsolverCsytrf_(handle,uplo,n,c_loc(A),lda,ipiv,work,lwork,devInfo)
    end function

    function hipsolverCsytrf_rank_0(handle,uplo,n,A,lda,ipiv,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCsytrf_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target :: A
      integer(c_int) :: lda
      type(c_ptr) :: ipiv
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCsytrf_rank_0 = hipsolverCsytrf_(handle,uplo,n,c_loc(A),lda,ipiv,work,lwork,devInfo)
    end function

    function hipsolverCsytrf_rank_1(handle,uplo,n,A,lda,ipiv,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCsytrf_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_float_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: ipiv
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverCsytrf_rank_1 = hipsolverCsytrf_(handle,uplo,n,c_loc(A),lda,ipiv,work,lwork,devInfo)
    end function

    function hipsolverZsytrf_full_rank(handle,uplo,n,A,lda,ipiv,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZsytrf_full_rank
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:,:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: ipiv
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZsytrf_full_rank = hipsolverZsytrf_(handle,uplo,n,c_loc(A),lda,ipiv,work,lwork,devInfo)
    end function

    function hipsolverZsytrf_rank_0(handle,uplo,n,A,lda,ipiv,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZsytrf_rank_0
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target :: A
      integer(c_int) :: lda
      type(c_ptr) :: ipiv
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZsytrf_rank_0 = hipsolverZsytrf_(handle,uplo,n,c_loc(A),lda,ipiv,work,lwork,devInfo)
    end function

    function hipsolverZsytrf_rank_1(handle,uplo,n,A,lda,ipiv,work,lwork,devInfo)
      use iso_c_binding
      use hipfort_hipsolver_enums
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZsytrf_rank_1
      type(c_ptr) :: handle
      integer(kind(HIPSOLVER_FILL_MODE_UPPER)) :: uplo
      integer(c_int) :: n
      complex(c_double_complex),target,dimension(:) :: A
      integer(c_int) :: lda
      type(c_ptr) :: ipiv
      type(c_ptr) :: work
      integer(c_int) :: lwork
      integer(c_int) :: devInfo
      !
      hipsolverZsytrf_rank_1 = hipsolverZsytrf_(handle,uplo,n,c_loc(A),lda,ipiv,work,lwork,devInfo)
    end function

  
#endif
end module hipfort_hipsolver