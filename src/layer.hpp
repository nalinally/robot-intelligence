#pragma once

#include <cstddef>
#include <cassert>
#include <vector>

namespace layer {

class Layer{

 public:  
  Layer();
  std::vector<double> getNodes();

 private:
  std::vector<double> nodes;

};

extern int tameshi;

}  // namespace layer