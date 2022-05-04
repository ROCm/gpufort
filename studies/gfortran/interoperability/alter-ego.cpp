#include <iostream>

template <int rank>
struct dope_vector {
  int extents[rank];
  int lbounds[rank];
  int operator() (int i1, int i2) {
    const int lb1   = this->lbounds[0];        
    const int lb2   = this->lbounds[1];        
    const int size1 = this->extents[0]; // size1 = stride2
    return 1*(i1-lb1) + size1 * (i2-lb2);
  }   
};

template <typename T, int rank, int size>
struct fixed_size_array {
  T              values[size]; 
  dope_vector<2> dope; 
  T& operator() (int i1, int i2) {
    return this->values[ this->dope(i1,i2) ];
  }   
};

namespace current_scope {
  constexpr int lb1 = -1; // lower bound dim 1 
  constexpr int ub1 =  2; // upper bound dim 1 
  constexpr int lb2 = -2; // lower bound dim 2 
  constexpr int ub2 =  3; // upper bound dim 2 
  constexpr int size1 = ub1-lb1+1;
  constexpr int size2 = ub2-lb2+1;
  constexpr int size = size1*size2;
}

extern "C" {
  using namespace current_scope; // need size constexpr
  void pass_to_cpp(fixed_size_array<int, 2, size>& array) {
    std::cout << "c++:" << std::endl;
    for (int i = lb1; i <= ub1; i++) {
      for (int j = lb2; j <= ub2; j++) {
        std::cout << "array("<<i<<","<<j<<")=" << array(i,j) << std::endl;
      }
    }
  }
}
