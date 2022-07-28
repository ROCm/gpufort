#include <iostream>

enum class record_creational_event {
  gpufortrt_event_undefined = 0,
  gpufortrt_event_create    = 1,
  gpufortrt_event_copyin    = 2,
  gpufortrt_event_copyout   = 3,
  gpufortrt_event_copy      = 4 
};

std::ostream& operator<<(std::ostream& os, record_creational_event ce)
{
    switch(ce)
    {
       case record_creational_event::gpufortrt_event_undefined : os << "undefined"; break;
       case record_creational_event::gpufortrt_event_create    : os << "create";    break;
       case record_creational_event::gpufortrt_event_copyin    : os << "copyin";    break;
       case record_creational_event::gpufortrt_event_copyout   : os << "copyout";   break;
       case record_creational_event::gpufortrt_event_copy      : os << "copy";      break;
       default: os.setstate(std::ios_base::failbit);
    }
    return os;
}
