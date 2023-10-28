import torch
import torchvision


class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)

    def forward(self, inputs):
        return self.resnet(inputs)


def main() -> None:
    # TODO: replace this with your code
    print("hello world")
    pass


if __name__ == "__main__":
    main()
