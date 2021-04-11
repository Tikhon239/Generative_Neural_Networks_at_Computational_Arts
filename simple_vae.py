import torch
torch.manual_seed(0)

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

class SimpleEncoder(nn.Module):
    def __init__(self, h, w, downsamplings=5, start_channels=8, latent_size=32):
        super().__init__()
        
        self.start_conv = nn.Conv2d(3, start_channels, 1, bias = False)
        
        self.downsamplings_conv = nn.Sequential(
            *[self.downsampling(start_channels * 2**i) for i in range(downsamplings)]
        )

        self.latent_size = latent_size
        self.end_conv =  nn.Conv2d(start_channels * 2 ** downsamplings, latent_size, 1, bias = False)
        
        self.end_fc = nn.Linear(latent_size * (h // 2 ** downsamplings) * (w // 2 ** downsamplings), latent_size)

    def downsampling(self, cur_channels):
        return nn.Sequential(
            nn.Conv2d(cur_channels, 2 * cur_channels, 3, 2, 1, bias = False),
            nn.BatchNorm2d(2 * cur_channels, affine = False),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.start_conv(x)
        x = self.downsamplings_conv(x)
        x = self.end_conv(x)
        
        x = x.flatten(start_dim=1)
        
        x = self.end_fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, h, w, upsamplings=5, start_channels=128, latent_size=32):
        super().__init__()
        self.h_start = (h // 2 ** upsamplings)
        self.w_start = (w // 2 ** upsamplings)
        self.latent_size = latent_size
        
        self.start_fc = nn.Linear(latent_size,  latent_size * self.h_start * self.w_start)
        
        self.start_conv = nn.Conv2d(latent_size, start_channels, 1)
        
        self.upsamplings_conv = nn.Sequential(
            *[self.upsampling(start_channels // 2 ** i) for i in range(upsamplings)]
        )
        
        self.end_conv = nn.Conv2d(start_channels // 2 ** upsamplings, 3, 1, bias = False)

    def upsampling(self, cur_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(cur_channels, cur_channels // 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(cur_channels // 2, affine = False),
            nn.ReLU()
        )
    
    def forward(self, z):
        x = F.relu(self.start_fc(z))
        
        x = self.start_conv(x.reshape(-1, self.latent_size, self.h_start, self.w_start))
        x = self.upsamplings_conv(x)
        x = self.end_conv(x)
        
        return torch.tanh(x)

class SimpleVAE(nn.Module):
    def __init__(self, h, w, downsamplings=5, latent_size=32, down_channels=8, up_channels=8):
        super().__init__()
        self.encoder = SimpleEncoder(h, w, downsamplings, down_channels, latent_size)
        self.decoder = Decoder(h, w, downsamplings, up_channels * 2 ** downsamplings, latent_size)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

def get_simple_vae(X, y, latent_size=128, batch_size=8, epochs = 100, device = 'cuda'):
  h, w = X.shape[-2:]

  dataset = TensorDataset(X, torch.Tensor(y))
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  simple_vae = SimpleVAE(h, w, downsamplings=5, latent_size=latent_size, down_channels=16, up_channels=16)

  simple_vae.to(device)

  total_batches = len(dataloader)

  simple_vae_optim = Adam(simple_vae.parameters(), lr=1e-4)

  for ep in range(epochs):
      reconstruction_loss_avg = 0
      for (x, _) in dataloader:
          x = x.to(device)
          x_hat = simple_vae(x)

          reconstruction_loss = ((x_hat - x)**2).mean()
          
          simple_vae_optim.zero_grad()
          reconstruction_loss.backward()
          simple_vae_optim.step()

          reconstruction_loss_avg += reconstruction_loss.item()

      print(f"Epoch {ep+1} | Reconstruction loss: {reconstruction_loss_avg / total_batches}")
  
  return simple_vae