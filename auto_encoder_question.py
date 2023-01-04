import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Function
import torchvision
import torch.nn.functional as F
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
import numpy as np

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
        
        # adjust last layer to cifar10
        self.resnet18.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.resnet18(x)


class Quantizer(nn.Module):
    def __init__(self):
        super(Quantizer, self).__init__()
        self.K = 1024    # Number of elements in dictionary
        self.D = 10   # Dimension of each element in dictionary
        
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.normal_()

        self.mse_loss = nn.MSELoss()

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
        
        # compute quantization loss
        # this loss is not backpropagated to the encoder network
        quantization_loss = self.mse_loss(encoder_embedding.detach(), quantized_embedding)
               
        # preserve gradients
        quantized_embedding = encoder_embedding + (quantized_embedding - encoder_embedding).detach()
        
        loss = quantization_loss
        
        return quantized_embedding, loss


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        # 10 ==> 64 ==> 128 ==> 256 ==> 512 ==> 1024 ==> 3072
        self.fc1 = nn.Linear(10,64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 32*32)
        self.fc6 = nn.Linear(1024, 32*32*3)
        
        self.dropout = nn.Dropout(0.2)       
        
    def forward(self, quantized_embedding):
        """
        Takes the quantized embedding from the quantizer network and reconstructs the image.
        """

        x = F.relu(self.fc1(quantized_embedding))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        x = F.relu(self.fc4(x))
        x = self.dropout(x)

        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        
        x = self.fc6(x)
        x = x.reshape(-1, 3, 32, 32)
        
        return x
        
# Training Loop

if __name__=="__main__":
    dataset = load_cifar10()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    EPOCHS = 50

    encoder = Encoder().to(device)
    quantizer = Quantizer().to(device)
    decoder = Decoder().to(device)

    mse_loss = nn.MSELoss()

    optimizer = torch.optim.SGD(list(encoder.parameters()) + list(quantizer.parameters()) + list(decoder.parameters()), lr=1e-1, weight_decay=1e-5)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[45], gamma=0.1)

    tb_writer = SummaryWriter()
    
    for epoch in list(range(EPOCHS)):
        
        batch_iter = tqdm(enumerate(dataloader), 'Training', total=len(dataloader), leave=False)
        train_losses = []
        for batch_idx, batch in batch_iter:
            input, target_label = batch
            input = input.to(device)
            # target = target.to(device)
            
            encoder_embedding = encoder(input)
            quantized_embedding, qaunt_loss = quantizer(encoder_embedding)
            pred = decoder(quantized_embedding)

            # reconstruction losses:
            recon_loss = mse_loss(input, pred)

            # total loss:
            loss = recon_loss + qaunt_loss
            
            optimizer.zero_grad()  # zerograd the parameters
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            batch_iter.set_description(f'Training: [{epoch:d}/{EPOCHS:d}] (loss {loss.item():.4f})')  # update progressbar

        
            if batch_idx % 10 == 0:               
                tb_writer.add_images('input', (input[:5,...]+0.5).clip(0,1), epoch)
                tb_writer.add_images('decoder output', (pred[:5,...]+0.5).clip(0,1), epoch)
                tb_writer.flush()
        
        tb_writer.add_scalar('Loss/train', np.mean(train_losses), epoch)
        tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        tb_writer.flush()
        lr_scheduler.step()

        
        
        
    