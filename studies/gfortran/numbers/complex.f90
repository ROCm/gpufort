program main 
   print *, "various integer and/or float components"
   print *, sizeof((1_4,2_4)) ! 8
   print *, sizeof((1.,2)) ! 8
   print *, sizeof((1.,2.)) ! 8
   print *, sizeof((1.,2_8)) ! 8
   print *, sizeof((1_8,2_8)) ! 8
   print *, sizeof((1_16,10000_16)) ! 8
   print *, "various integer and/or double components"
   print *, sizeof((1.,2.0_8)) ! 16
   print *, sizeof((1._8,2.0_8)) ! 16
   print *, "one quad component"
   print *, sizeof((1._8,2.0_16)) ! 32
   print *, sizeof((1._16,2.0_8)) ! 32
   print *, sizeof((1._16,2.0_16)) ! 32
   ! not legal to have logical components
   !print *, (1.,.true.)
end program        
