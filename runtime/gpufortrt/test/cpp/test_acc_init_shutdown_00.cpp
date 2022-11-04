#include <openacc.h>

int main( int argc, char * argv[] )
{
   acc_init( acc_device_default );

   float a[100];

   #pragma acc data copyout(a[0:100])

   #pragma acc parallel
   #pragma acc loop
   for( int i = 0; i < 100; ++i )
   {
      a[i] = 5;
   }

   acc_shutdown( acc_device_default );

   return 0;
}