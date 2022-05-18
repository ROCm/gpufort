module mymod

type mytype
private
  integer :: scalar
end type

public operator(.eq.), assignment(=)

end module
