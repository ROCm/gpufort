program main
  if( tha(i,j,k).gt. 1.0 ) thrad = -1.0/(12.0*3600.0)
  WHERE ( ht_g(ids:ide,jds:jde) < -1000. ) ht_g(ids:ide,jds:jde) = 0
end program
