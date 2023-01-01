# %%
import comet_ml

# %%
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger

import torch
from torch import nn
from torch.nn import functional as F

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

    def __init__(self, latent_width: int = 64, **kwargs) -> None:
        super().__init__()
        self.latent_width = latent_width

        self.model = nn.Sequential(
            Encoder.conv_block(1, 8),
            Encoder.conv_block(8, 8),
            Encoder.conv_block(8, 16),
            Encoder.conv_block(16, 16),
        )

        self.fc_mu = nn.Linear(16 * 2 * 2, self.latent_width)
        self.fc_logvar = nn.Linear(16 * 2 * 2, self.latent_width)

    def reparametrize(self, mean: torch.Tensor, logvariance: torch.Tensor):
        # transform N(0,1) into N(mean, variance)
        std_dev = torch.exp(0.5 * logvariance)
        eps = torch.randn_like(std_dev)
        return eps * std_dev + mean

    def encode(self, x: torch.Tensor):
        x = self.model(x)
        x = einops.rearrange(x, "b c w h -> b (c w h)")
        mean = self.fc_mu(x)
        logvariance = self.fc_logvar(x)
        return mean, logvariance

    def forward(self, x: torch.Tensor):
        mean, logvariance = self.encode(x)
        z = self.reparametrize(mean, logvariance)
        return z, mean, logvariance


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

    def __init__(self, latent_width: int = 64, **kwargs) -> None:
        super().__init__()
        self.latent_width = latent_width

        # self.fc = nn.Sequential(
        #     nn.Linear(self.latent_width, 16 * 2 * 2),
        #     nn.ReLU(),
        # )
        self.fc = nn.Linear(self.latent_width, 16 * 2 * 2)
        self.model = nn.Sequential(
            Decoder.conv_block(16, 16),
            Decoder.conv_block(16, 8),
            Decoder.conv_block(8, 8),
            Decoder.conv_block(8, 8),
            nn.Conv2d(8, 1, kernel_size=5, stride=1, padding=0),
            nn.Sigmoid(),  # end result is an image, it must be in [0, 1] range!
        )

    def forward(self, x: torch.Tensor):
        x = self.fc(x)
        # x = torch.reshape(x, (-1, 16, 2, 2))
        x = einops.rearrange(x, "b (c w h) -> b c w h", c=16, w=2, h=2)
        x = self.model(x)
        return x


# %%
class Autoencoder(pl.LightningModule):
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
        latent, mean, logvariance = self.encoder(images)
        decoded = self.decoder(latent)

        return decoded, latent, mean, logvariance

    def reconstruction_loss(self, predictions, targets):
        return F.mse_loss(predictions, targets)

    def kld_loss(self, mean, logvariance):
        # distance from normal distribution
        loss = einops.reduce(
            1 + logvariance - mean**2 - torch.exp(logvariance),
            "batch latent -> batch",
            "sum",
        )
        return einops.reduce(-0.5 * loss, "batch -> ()", "mean")

        # commonly seen version:
        # return torch.mean(
        #     -0.5 * torch.sum(1 + logvariance - mean**2 - logvariance.exp(), dim=1),
        #     dim=0,
        # )

    # NOT standard in PyTorchLightning
    def process_step(self, batch, batch_idx):
        images, _ = batch
        predictions, latents, mean, logvariance = self(images)

        reconstruction = self.reconstruction_loss(predictions, images)
        kld = self.kld_loss(mean, logvariance)
        loss = reconstruction + 0.01 * kld

        self.log("loss", loss, on_epoch=True)
        self.logger.experiment.log_metric("loss", loss)
        self.logger.experiment.log_metric("loss_reconstruction", reconstruction)
        self.logger.experiment.log_metric("loss_kld", kld)

        return loss, images, predictions, latents

    def training_step(self, train_batch, batch_idx):
        with self.logger.experiment.train():
            loss, _, _, latents = self.process_step(train_batch, batch_idx)
        self.logger.experiment.log_histogram_3d(
            latents.detach().cpu(),
            "latents",
            step=self.global_step,
        )

        return loss

    @torch.no_grad()
    def validation_step(self, val_batch, batch_idx):
        with self.logger.experiment.validate():
            _, images, predictions, _ = self.process_step(val_batch, batch_idx)
            images = einops.rearrange(
                [images, predictions],
                "source batch 1 height width -> batch height (source width)",
            ).cpu()

            # self.logger.experiment.log_image(images[0], f"images_{batch_idx}")

    @torch.no_grad()
    def test_step(self, test_batch, batch_idx):
        with self.logger.experiment.test():
            _, images, predictions, _ = self.process_step(test_batch, batch_idx)
            images = einops.rearrange(
                [images, predictions],
                "source batch 1 height width -> (batch height) (source width)",
            ).cpu()

            self.logger.experiment.log_image(images, f"images_{batch_idx}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# %%
import os

from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms

# from torchinfo import summary

if __name__ == "__main__":
    # data
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

    # model
    latent_width = 64

    encoder = Encoder(latent_width)
    decoder = Decoder(encoder.latent_width)
    model = Autoencoder(encoder, decoder)
    model.cuda()

    # logging
    comet_logger = CometLogger(project_name="UczenieNienadzorowane")
    comet_logger.log_graph(model)
    # comet_logger.experiment.set_model_graph(model.__repr__())
    comet_logger.experiment.log_code(folder="UN")
    comet_logger.experiment.log_parameters(
        {
            "batch_size": train_loader.batch_size,
            "train_size": len(mnist_train),
            "val_size": len(mnist_val),
        }
    )
    # summ = summary(model, (1, 1, 32, 32), device="cuda", depth=5)
    # comet_logger.experiment.set_model_graph(f"{model.__repr__()}\n{summ}")
    comet_logger.experiment.log_parameter("loss_func", "VAE loss")

    comet_logger.experiment.add_tag("LOSS: VAE loss")

    # training
    trainer = pl.Trainer(max_epochs=5, gpus=1, precision=16, logger=comet_logger)
    comet_logger.experiment.log_parameter("num_epochs", trainer.max_epochs)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, val_loader)

    # Log model
    model_name = f"{type(model).__name__}_{comet_logger.experiment.get_key()}"
    torch.save(model.state_dict(), f"{model_name}.pth")
    comet_logger.experiment.log_model(model_name, f"./{model_name}.pth")
    os.remove(f"./{model_name}.pth")
