#include <iostream>

extern "C" {

  typedef struct b {
    int mb;
  } b;
  
  typedef struct a {
    b tb;
  } a;
  
  void read_nested_struct(a* ta) {
    std::cout << ((*ta).tb.mb) << std::endl;
  }

}
