import torch
import einops
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    @staticmethod
    def conv_block(in_size: int, out_size: int):
        return nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(),
        )

    def __init__(self, latent_width: int = 64, **kwargs) -> None:
        super().__init__()
        self.latent_width = latent_width

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
        x = self.model(x)
        # x = torch.flatten(x, start_dim=1)
        x = einops.rearrange(x, "b c w h -> b (c w h)")
        x = self.fc(x)
        return x


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
                bias=False
            ),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(),
        )

    def __init__(self, latent_width: int = 64, **kwargs) -> None:
        super().__init__()
        self.latent_width = latent_width

        self.fc = nn.Sequential(
            nn.Linear(self.latent_width, 16 * 2 * 2),
            nn.ReLU(),
        )
        self.model = nn.Sequential(
            Decoder.conv_block(16, 16),
            Decoder.conv_block(16, 8),
            Decoder.conv_block(8, 8),
            Decoder.conv_block(8, 8),
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),  # end result is an image, it must be in [0, 1] range!
        )

    def forward(self, x: torch.Tensor):
        x = self.fc(x)
        # x = torch.reshape(x, (-1, 16, 2, 2))
        x = einops.rearrange(x, "b (c w h) -> b c w h", c=16, w=2, h=2)
        x = self.model(x)

        return x


class Autoencoder(nn.Module):
    def __init__(
            self,
            encoder: Encoder,
            decoder: Decoder,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images: torch.Tensor):
        latent = self.encoder(images)
        decoded = self.decoder(latent)

        return decoded, latent
