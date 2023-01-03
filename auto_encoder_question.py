import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Function
import torchvision
import torch.nn.functional as F

def load_cifar10():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = torchvision.datasets.CIFAR10('.', download=True, transform=transform)
    return dataset


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        

    def forward(self, x):
        return self.resnet18(x)


class Quantizer(nn.Module):
    def __init__(self):
        super(Quantizer, self).__init__()
        self.K = 256    # Number of elements in dictionary
        self.D = 1000   # Dimension of each element in dictionary
        
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1/self.K, 1/self.D)

    def forward(self, encoder_embedding):
        """
        Takes the input from the encoder network and finds the nearest tensor in the dictionary.
        This Tensor will later be passed to the decoder network.
        """
        
        # calculate the distance between the encoder embedding and every element in the dictionary
        distance = torch.sum((encoder_embedding.unsqueeze(1) - self.embedding.weight)**2, dim=2)
        
        # find the index of the nearest element in the dictionary
        quantized_embedding_indices = torch.argmin(distance, dim=1)

        # find the nearest element in the dictionary
        quantized_embedding = self.embedding(quantized_embedding_indices)
        
        
        return quantized_embedding
        


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        # number of hidden nodes in each layer (512-256-128-64)
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 =  nn.Linear(512, 768)
        self.fc3 =  nn.Linear(768, 1024)

        self.dropout = nn.Dropout(0.2)       
        
    def forward(self, quantized_embedding):
        """
        Takes the quantized embedding from the quantizer network and reconstructs the image.
        """

        x = F.relu(self.fc1(quantized_embedding))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = F.linear(self.fc3(x))
        
        return x
        



# Training Loop

if __name__=="__main__":
    dataset = load_cifar10()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    encoder = Encoder()
    quantizer = Quantizer()
    decoder = Decoder()

    for batch in dataloader:
        input, target = batch
        encoder_embedding = encoder(input)
        quantized_embedding = quantizer(encoder_embedding)
        reconstructed_batch = decoder(quantized_embedding)

        # Compute loss and backpropagate

        # Update weights

    pass