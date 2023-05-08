program main
!$acc data copyin(a,b) copyout(c_gpu)
!$acc parallel loop collapse(2) &
!$acc reduction(+:tmp)
do j=1,colsB
do i=1,rowsA
tmp = 0.0
!$acc loop vector reduction(+:tmp)
do k=1,rowsB
tmp = tmp &
+ a(i,k) * b(k,j)
enddo
c_gpu(i,j) = tmp
enddo
enddo
!$acc end parallel
!$acc end data
end program
