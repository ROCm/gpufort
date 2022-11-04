#include <openacc.h>

#pragma acc routine worker
void set( int* in_out )
{
   *in_out = ( *in_out ) * 3;
}

int main( int argc, char * argv[] )
{
   acc_init( acc_device_default );

   float a[100];

#pragma acc data copyout(a[0:100])

#pragma acc parallel
#pragma acc loop
   for( int i = 0; i < 100; ++i )
   {
      int j = 5;
      set(&j);
      a[i] = j;
   }

   acc_shutdown( acc_device_default );

   return 0;
}