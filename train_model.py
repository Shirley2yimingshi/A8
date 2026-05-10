# train_model.py

import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {device}\n")


LATENT_DIM = 2

BATCH_SIZE = 128

EPOCHS = 15

LR = 1e-3

BETA = 0.001

DATASET_NAME = "MNIST"


# 数据集

transform = transforms.ToTensor()

if DATASET_NAME == "MNIST":

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

else:

    train_dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print(f"Dataset Loaded: {DATASET_NAME}")
print(f"Training Samples: {len(train_dataset)}")


# 创建 checkpoint 文件夹
os.makedirs(
    "checkpoints",
    exist_ok=True
)

# AutoEncoder
class AutoEncoder(nn.Module):

    def __init__(self, latent_dim=2):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(784, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, latent_dim)
        )

        self.decoder = nn.Sequential(

            nn.Linear(latent_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.ReLU(),

            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def forward(self, x):

        z = self.encoder(x)

        out = self.decoder(z)

        return out

# VAE
class VAE(nn.Module):

    def __init__(self, latent_dim=2):

        super().__init__()

        # Encoder
        self.fc1 = nn.Linear(784, 512)

        self.fc2 = nn.Linear(512, 256)

        self.fc_mu = nn.Linear(256, latent_dim)

        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 256)

        self.fc4 = nn.Linear(256, 512)

        self.fc5 = nn.Linear(512, 784)

    def encode(self, x):

        h = torch.relu(self.fc1(x))

        h = torch.relu(self.fc2(h))

        mu = self.fc_mu(h)

        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z):

        h = torch.relu(self.fc3(z))

        h = torch.relu(self.fc4(h))

        out = torch.sigmoid(self.fc5(h))

        return out

    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        out = self.decode(z)

        return out, mu, logvar

# Generator
class Generator(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(100, 256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.ReLU(),

            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z):

        return self.model(z)

# =========================================================
# Discriminator
# =========================================================

class Discriminator(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        return self.model(x)

# 初始化模型
ae = AutoEncoder(LATENT_DIM).to(device)

vae = VAE(LATENT_DIM).to(device)

generator = Generator().to(device)

discriminator = Discriminator().to(device)

# Loss
mse_loss = nn.MSELoss()

bce_loss = nn.BCELoss()

# Optimizer
ae_optimizer = optim.Adam(
    ae.parameters(),
    lr=LR
)

vae_optimizer = optim.Adam(
    vae.parameters(),
    lr=LR
)

g_optimizer = optim.Adam(
    generator.parameters(),
    lr=0.0002
)

d_optimizer = optim.Adam(
    discriminator.parameters(),
    lr=0.0002
)

# Train AutoEncoder
print("\n==============================")
print("Training AutoEncoder...")
print("==============================\n")

ae.train()

for epoch in range(EPOCHS):

    epoch_loss = 0

    for images, _ in train_loader:

        images = images.view(-1, 784).to(device)

        outputs = ae(images)

        loss = mse_loss(
            outputs,
            images
        )

        ae_optimizer.zero_grad()

        loss.backward()

        ae_optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)

    print(
        f"AE Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {avg_loss:.6f}"
    )

# 保存 AE
torch.save(

    ae.state_dict(),

    "checkpoints/ae.pth"
)

print("\n✅ AE model saved!")

# Train VAE
print("\n==============================")
print("Training VAE...")
print("==============================\n")

vae.train()

for epoch in range(EPOCHS):

    epoch_loss = 0

    recon_epoch = 0

    kl_epoch = 0

    for images, _ in train_loader:

        images = images.view(-1, 784).to(device)

        recon, mu, logvar = vae(images)

        # BCE Reconstruction Loss
        recon_loss = F.binary_cross_entropy(

            recon,

            images,

            reduction='sum'

        ) / images.shape[0]

        # KL Divergence
        kl_loss = -0.5 * torch.sum(

            1 + logvar - mu.pow(2) - logvar.exp()

        ) / images.shape[0]

        # Final Loss
        loss = recon_loss + BETA * kl_loss

        vae_optimizer.zero_grad()

        loss.backward()

        vae_optimizer.step()

        epoch_loss += loss.item()

        recon_epoch += recon_loss.item()

        kl_epoch += kl_loss.item()

    avg_loss = epoch_loss / len(train_loader)

    avg_recon = recon_epoch / len(train_loader)

    avg_kl = kl_epoch / len(train_loader)

    print(
        f"VAE Epoch [{epoch+1}/{EPOCHS}] "
        f"Total: {avg_loss:.4f} | "
        f"Recon: {avg_recon:.4f} | "
        f"KL: {avg_kl:.4f}"
    )

# 保存 VAE
torch.save(

    vae.state_dict(),

    "checkpoints/vae.pth"
)

print("\n✅ VAE model saved!")

# Train GAN
print("\n==============================")
print("Training DCGAN...")
print("==============================\n")

generator.train()

discriminator.train()

for epoch in range(EPOCHS):

    g_epoch_loss = 0

    d_epoch_loss = 0

    for images, _ in train_loader:

        batch_size_now = images.shape[0]

        real_images = images.view(
            batch_size_now,
            -1
        ).to(device)

        valid = torch.ones(
            batch_size_now,
            1
        ).to(device)

        fake = torch.zeros(
            batch_size_now,
            1
        ).to(device)

        # Train Generator
        z = torch.randn(
            batch_size_now,
            100
        ).to(device)

        generated_images = generator(z)

        g_loss = bce_loss(

            discriminator(generated_images),

            valid
        )

        g_optimizer.zero_grad()

        g_loss.backward()

        g_optimizer.step()

        # Train Discriminator
        real_loss = bce_loss(

            discriminator(real_images),

            valid
        )

        fake_loss = bce_loss(

            discriminator(
                generated_images.detach()
            ),

            fake
        )

        d_loss = (
            real_loss + fake_loss
        ) / 2

        d_optimizer.zero_grad()

        d_loss.backward()

        d_optimizer.step()

        g_epoch_loss += g_loss.item()

        d_epoch_loss += d_loss.item()

    avg_g_loss = g_epoch_loss / len(train_loader)

    avg_d_loss = d_epoch_loss / len(train_loader)

    print(
        f"GAN Epoch [{epoch+1}/{EPOCHS}] "
        f"G Loss: {avg_g_loss:.6f} "
        f"D Loss: {avg_d_loss:.6f}"
    )

# 保存 GAN
torch.save(

    generator.state_dict(),

    "checkpoints/generator.pth"
)

print("\n✅ GAN Generator saved!")


# 完成
print("\n======================================")
print("🎉 ALL MODELS TRAINED SUCCESSFULLY!")
print("======================================")

print("\nSaved files:")

print("checkpoints/ae.pth")

print("checkpoints/vae.pth")

print("checkpoints/generator.pth\n")