#include <iostream>

int main(int argc, char** argv) {
  std::cout << "scenario 3 (C++ implementation):" << std::endl;
  for (int i=1; i<=3; i++) {
    std::cout << "start inner" << std::endl;
    for (int j=1; j<=10; j++) {
      std::cout << "i,j="<< i << "," << j << std::endl;
      if ( j == 1 ) {
        std::cout << "continue as j == 1" << std::endl;
        /*continue*/; // expect j = 2 afterwards;
        std::cout << "continue is C++ empty statement, so this will be printed" << std::endl;
      } else if ( j == 3 ) {
        std::cout << "cycle as j == 3" << std::endl;
        /*cycle*/continue; // expect j = 4 afterwards;
        std::cout << "cycle is C++ continue, so this will not be printed" << std::endl;
      } else if ( j == 5 ) {
        std::cout << "exit as j == 5" << std::endl;
        /*exit*/break;;
        std::cout << "exit is C++ break, so this will not be printed" << std::endl;
      }
    }
  }
  std::cout << "end outer" << std::endl;
}
