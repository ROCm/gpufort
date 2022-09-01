program test0025
  !$acc serial
  SNOALB2=SNOALB1*(SNACCA**((SNOTIME1/86400.0)**SNACCB))
  !$acc end serial
end program
