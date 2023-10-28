#include <iostream>
#include <vector>

#include <torch/script.h>

int main(int argc, char *argv[]) {
  // Run on CPU
  torch::Device device(torch::kCPU);

  torch::jit::script::Module model;
  try {
    // Should read the path from a config file instead of hardcoding
    model = torch::jit::load("./../resnet18_model.pt");
  }
  catch (const c10::Error& e) {
    std::cerr << "Error loading the model\n";
    return -1;
  }

  model.to(device);
  model.eval();

  // Example input tensor
  auto tensor = torch::randn({1, 3, 224, 224}).to(device);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(tensor);

  auto output = model.forward(inputs).toTensor();
  std::cout<<output.slice(1, 0, 5)<<"\n";

  std::cout<<"Ok\n";
}
