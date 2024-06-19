import torch


def get_conv_output_size(size):
    size = (size - 11) // 4 + 1
    size = (size - 5) // 2 + 1
    size = (size - 5) // 2 + 1
    size = (size - 5) // 2 + 1
    size = (size - 8) // 1 + 1
    return 400 * size * size


def reparameterise(mu, sigma):
    std = torch.exp(0.5*sigma)
    eps = torch.randn_like(std)
    z = mu + eps*std
    return z
