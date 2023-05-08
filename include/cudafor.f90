
module cudafor
  type,bind(c) :: dim3
     integer(c_int) :: x,y,z
  end type dim3

  type(dim3) :: blockdim, griddim, threadidx, blockidx
end module cudafor