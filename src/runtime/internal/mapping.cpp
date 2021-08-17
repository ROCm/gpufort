#include "internal/mapping.h"

bool internal::contains_subregion(char* region, std::size_t bytes, char* subregion, std::size_t bytes_subregion) {
  std::ptrdiff_t relative_offset = subregion - region;
  return (subregion >= region) && (relative_offset < bytes) && ((subregion+bytes_subregion) <= region+bytes);    
}
