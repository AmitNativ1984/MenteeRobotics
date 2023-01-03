import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Function
import torchvision


def load_cifar10():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = torchvision.datasets.CIFAR10('.', download=True, transform=transform)
    return dataset


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        pass

    def forward(self, x):
        pass


class Quantizer(nn.Module):
    def __init__(self):
        super(Quantizer, self).__init__()
        pass

    def forward(self, encoder_embedding):
        pass


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        pass

    def forward(self, quantized_embedding):
        pass


# Training Loop