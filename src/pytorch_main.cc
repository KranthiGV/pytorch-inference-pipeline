#include <iostream>
#include <vector>

#include <torch/script.h>

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: pytorch_main <path-to-exported-script-module>\n";
    return -1;
  }

  // Run on CPU
  torch::Device device(torch::kCPU);

  torch::jit::script::Module model;
  try {
    model = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "Error loading the model\n";
    return -1;
  }

  torch::NoGradGuard no_grad;
  model.to(device);
  // Disable any dropout or BN layers
  model.eval();

  // Example input tensor
  auto tensor = torch::randn({1, 3, 224, 224}).to(device);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(tensor);

  auto output = model.forward(inputs).toTensor();
  std::cout<<"Ran inference on sample input\n";
  std::cout<<output.slice(1, 0, 5)<<"\n";
}
