# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch.nn import functional as F

import einops

# %%
class Encoder(nn.Module):
    @staticmethod
    def conv_block(in_size: int, out_size: int):
        return nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(),
        )

    def __init__(self, latent_width: int = 64) -> None:
        super().__init__()
        self.latent_width = latent_width

        # self.cnn1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)
        # self.act1 = nn.LeakyReLU()
        # self.cnn2 = nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1)
        # self.act2 = nn.LeakyReLU()

        # self.model = nn.Sequential(
        #     nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(8),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(8),
        #     nn.LeakyReLU(),
        # )

        self.model = nn.Sequential(
            Encoder.conv_block(1, 8),
            Encoder.conv_block(8, 8),
            Encoder.conv_block(8, 16),
            Encoder.conv_block(16, 16),
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 2 * 2, self.latent_width),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        # x = self.cnn1(x)
        # x = self.act1(x)
        # x = self.cnn2(x)
        # x = self.act2(x)
        x = self.model(x)
        # x = torch.flatten(x, start_dim=1)
        x = einops.rearrange(x, "b c w h -> b (c w h)")
        x = self.fc(x)

        return x


# %%
class Decoder(nn.Module):
    @staticmethod
    def conv_block(in_size: int, out_size: int):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_size,
                out_size,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(),
        )

    def __init__(self, latent_width: int = 64) -> None:
        super().__init__()
        self.latent_width = latent_width

        self.fc = nn.Sequential(
            nn.Linear(self.latent_width, 16 * 2 * 2),
            nn.Tanh(),
        )

        self.model = nn.Sequential(
            Decoder.conv_block(16, 16),
            Decoder.conv_block(16, 8),
            Decoder.conv_block(8, 8),
            Decoder.conv_block(8, 8),
            nn.Conv2d(8, 1, kernel_size=5, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        x = self.fc(x)
        x = einops.rearrange(x, "b (c w h )-> b c w h", c=16, w=2, h=2)
        x = self.model(x)

        return x


# %%
class Autoencoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return decoded, latent


# %%
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from torchinfo import summary

from tqdm import tqdm

if __name__ == "__main__":
    # Dane:
    mnist_train = MNIST(
        "/datasets",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    mnist_val = MNIST(
        "/datasets",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    train_loader = DataLoader(
        mnist_train,
        batch_size=32,
        num_workers=32,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        mnist_val,
        batch_size=32,
        num_workers=32,
        shuffle=False,
        pin_memory=True,
    )

    # Model:
    latent_width = 64
    encoder = Encoder(latent_width)
    decoder = Decoder(encoder.latent_width)
    model = Autoencoder(encoder, decoder)
    model.cuda()

    # Training:
    num_epochs = 5
    loss_func = F.mse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(num_epochs):
        model.eval()
        with torch.no_grad():
            for images, _ in tqdm(val_loader):
                images = images.cuda()
                predictions, latents = model(images)
                loss = loss_func(predictions, images)
                print(loss)

        model.train()
        for images, _ in tqdm(train_loader):

            optimizer.zero_grad()
            images = images.cuda()
            predictions, latents = model(images)
            loss = loss_func(predictions, images)
            loss.backward()
            optimizer.step()
            print(loss)

# %%
