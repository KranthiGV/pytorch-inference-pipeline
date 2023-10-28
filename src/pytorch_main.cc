#include <iostream>

#include <torch/torch.h>

int main(int argc, char *argv[]) {
  // TODO: replace this with your code
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
  std::cout << "finished fine!" << std::endl;
}
