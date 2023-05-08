module private_mod1
  private

  integer private_var1
  integer,private :: private_var2
  integer :: private_var2

  integer,public :: public_var1
  integer public_var2

  type :: private_type1
  end type
  
  type,public :: public_type1
  end type
  
  type :: public_type2
  end type

  private :: private_var2
  
  public :: public_var2, public_type2, public_proc1
contains

  subroutine private_proc1
  end subroutine 
  
  subroutine public_proc1
  end subroutine 
end module

module public_mod1
  public

  integer public_var1
  integer,public :: public_var2
  integer :: public_var2

  integer,private :: private_var1
  integer private_var2

  type :: public_type1
  end type
  
  type,private :: private_type1
  end type
  
  type :: private_type2
  end type

  public :: public_var2
  
  private :: private_var2, private_type2, private_proc1
contains

  subroutine public_proc1
  end subroutine 
  
  subroutine private_proc1
  end subroutine 
end module

module public_mod2
  !public ! public is the default accessibility

  integer public_var1
  integer,public :: public_var2
  integer :: public_var2

  integer,private :: private_var1
  integer private_var2

  type :: public_type1
  end type
  
  type,private :: private_type1
  end type
  
  type :: private_type2
  end type

  public :: public_var2
  
  private :: private_var2, private_type2, private_proc1
contains

  subroutine public_proc1
  end subroutine 
  
  subroutine private_proc1
  end subroutine 
end module
