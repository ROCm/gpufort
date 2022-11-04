#include <openacc.h>
#include <iostream>

int main( int argc, char * argv[] )
{
   acc_device_t t = acc_device_default;
   std::cout<< "number of devices: " << acc_get_num_devices(t);

   return 0;
}