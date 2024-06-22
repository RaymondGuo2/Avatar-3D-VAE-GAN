import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle


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


# Function taken from Bryon Kucharski (see ReadMe file)
def SavePloat_Voxels(voxels, path, iteration):
    voxels = voxels[:8].__ge__(0.5)
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')

    plt.savefig(path + '/{}.jpg'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.close()

    with open(path + '/{}.pkl'.format(str(iteration).zfill(3)), "wb") as f:
        pickle.dump(voxels, f, protocol=pickle.HIGHEST_PROTOCOL)


