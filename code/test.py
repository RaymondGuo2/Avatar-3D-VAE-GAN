import torch
from torch import optim
from model import Encoder, Generator, Discriminator
from torch.utils.data import DataLoader
from utils import reparameterise, SavePloat_Voxels
import torch.autograd
import os


def test_vae(dataset, batch_size=32, lr=3e-4):

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

    recon_loss_total = 0
    for i, (image, volume) in enumerate(dset):
        # Deal with values smaller than batch size
        if volume.size()[0] != int(batch_size):
            continue

        image = image.to(device).permute(0, 3, 1, 2).float()
        volume = volume.to(device).float()

        # Pass the image through the encoder and generate the latent
        z_mu, z_var = encoder(image)
        z_vae = reparameterise(z_mu, z_var)
        g_vae = generator(z_vae)

        # Calculate reconstruction loss (taken from Kucharski (see ReadMe))
        recon_loss = torch.sum(torch.pow((g_vae - volume), 2), dim=(1, 2, 3))
        print(recon_loss.size())
        print("RECON LOSS ITER: " ,i," - ", torch.mean(recon_loss))
        recon_loss_total += recon_loss
        samples = g_vae.cpu().data[:8].squeeze().numpy()

        image_path = "../data/3DVAEGAN_test"
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        SavePloat_Voxels(samples, image_path, i)

