#include <iostream>

extern "C" {

  typedef struct {
    int mb;
  } b;
  
  typedef struct {
    b tb;
    b tb_static_array[5];
    b* tb_dyn_array;
    int tb_dyn_array_n1;
    int tb_dyn_array_lb1;
  } a;
  
  void read_nested_struct(a* ta) {
    std::cout << (ta->tb.mb) << std::endl;
    std::cout << (ta->tb_dyn_array) << std::endl;
    std::cout << (ta->tb_dyn_array_n1) << std::endl;
    std::cout << (ta->tb_dyn_array_lb1) << std::endl;

    for (int i = 0; i < ta->tb_dyn_array_n1; i++) {
      std::cout << ta->tb_dyn_array[i].mb << std::endl;
    }
  }

}
