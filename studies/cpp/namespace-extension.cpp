#include <iostream>

using namespace std;

namespace ns {
    constexpr int a = 5;
}

namespace ns {
  struct t {
    int arr[a];
  };
}
typedef ns::t t;

int main() {
    t mytype;
    
    return 0;
}
