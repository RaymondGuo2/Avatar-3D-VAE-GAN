import torch
from torch import nn, optim
from model import Encoder, Generator, Discriminator
from torch.utils.data import DataLoader
from utils import reparameterise
import torch.autograd


# torch.autograd.set_detect_anomaly(True)


def train_vae(dataset, epochs=100, batch_size=32, lr=3e-4):

    # Load the dataset
    dset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    encoder = Encoder()
    generator = Generator()
    discriminator = Discriminator()

    # Move models to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    generator.to(device)
    discriminator.to(device)

    # Optimizers
    optimizer_e = optim.Adam(encoder.parameters(), lr=lr)
    optimizer_g = optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

    # Loss functions
    bce_loss = nn.BCELoss()

    for epoch in range(epochs):
        for i, (image, volume) in enumerate(dset):

            if volume.size()[0] != int(batch_size):
                continue

            image = image.to(device).permute(0, 3, 1, 2).float()
            volume = volume.to(device).float()

            # Pass the image through the encoder and generate the latent
            z_mu, z_var = encoder(image)
            z_vae = reparameterise(z_mu, z_var)

            # Forward pass of real 3D object through discriminator and compare loss to real labels
            d_real = discriminator(volume)
            real_labels = torch.ones_like(discriminator(volume)).to(device).float()
            d_real_loss = bce_loss(d_real, real_labels)

            # Fake samples
            fake = generator(z_vae.detach())
            d_fake = discriminator(fake)
            fake_labels = torch.zeros_like(d_fake).to(device).float()
            d_fake_loss = bce_loss(d_fake, fake_labels)

            # Discriminator loss and gradient descent
            d_loss = d_real_loss + d_fake_loss
            optimizer_d.zero_grad()
            d_loss.backward(retain_graph=True)
            optimizer_d.step()

            # Calculate discriminator accuracy
            d_real_acu = torch.ge(d_real.squeeze(), 0.5).float()
            d_fake_acu = torch.le(d_fake.squeeze(), 0.5).float()
            d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu), 0))

            # Reconstruction and KL (Encoder) Loss
            volume = volume.view(-1, 1, 64, 64, 64)
            recon_loss = torch.sum(torch.pow((fake - volume), 2))
            kl_loss = (- 0.5 * torch.sum(1 + z_var - torch.pow(z_mu, 2) - torch.exp(z_var)))
            e_loss = recon_loss + kl_loss

            # Encoder gradient descent
            optimizer_e.zero_grad()
            e_loss.backward(retain_graph=True)
            optimizer_e.step()

            # Generator loss function
            d_fake = discriminator(fake)
            g_loss = bce_loss(d_fake, real_labels)
            g_loss = g_loss + recon_loss

            # Generator gradient descent
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            print('Epoch [{} / {}], Step [{} / {}], Recon Loss: {:.4f}, KL Loss: {:.4f}, D Loss: {:.4f}, G Loss: {:.4f}, D Acc: {:.4f}'.format(
                epoch + 1, epochs, i + 1, len(dset), recon_loss.item(), kl_loss.item(), d_loss.item(), g_loss.item(), d_total_acu.item()))
