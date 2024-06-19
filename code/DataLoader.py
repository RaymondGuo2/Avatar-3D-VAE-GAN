import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import torch


# Load the dataset
class DataLoader(Dataset):
    def __init__(self, root_dir, image_size=28):
        self.root = root_dir
        self.dir = os.listdir(self.root)
        self.image_size = image_size
        self.resize = transforms.Compose([transforms.Resize((self.image_size, self.image_size))])

    def __getitem__(self, index):
        # Search and load the 3D numpy array file
        volume_file = [name for name in self.dir if name.endswith('.' + "npy")][index]
        volume = np.load(os.path.join(self.root, volume_file)).astype(np.bool_)
        # Identify associated image file
        image_jpg = volume_file.replace("_mesh.npy", ".jpg")
        image = Image.open(self.root + "/" + image_jpg)
        # Represent the image as a fixed size 2D array
        image = np.asarray(self.resize(image))
        print(image.shape, volume.shape)
        print(volume_file, image_jpg)
        # Convert to tensors
        return torch.FloatTensor(image.copy()), torch.BoolTensor(volume)

    def __len__(self):
        return len([name for name in self.dir if name.endswith('.' + "npy")])

if __name__ == '__main__':
    data = DataLoader(root_dir="../data/", image_size=28)
    print(data[5])