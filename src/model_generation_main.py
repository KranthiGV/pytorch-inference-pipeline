import torch
import torchvision
from torchvision.models import ResNet18_Weights

class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)

    def forward(self, inputs):
        return self.resnet(inputs)


def main() -> None:
    model = TestModel()

    example_input = torch.rand(1, 3, 224, 224)
    model_traced = torch.jit.trace(model, example_input)
    model_traced.save("resnet18_model.pt")


if __name__ == "__main__":
    main()
