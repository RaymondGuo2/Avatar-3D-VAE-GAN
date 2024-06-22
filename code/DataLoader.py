import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import random
import shutil


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


# Function to store corresponding .jpg, .obj, .npy, and .binvox files as sets
def split_files_to_sets(path):
    # Create dictionary to store sets of files
    file_set = {}
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            base = os.path.splitext(file)[0]
            binvox_path = file.replace(".jpg", "_mesh.binvox")
            obj_path = file.replace(".jpg", "_mesh.obj")
            numpy_path = file.replace(".jpg", "_mesh.npy")
            # Store sets of files together
            file_set[base] = {'jpg': file, 'binvox': binvox_path, 'obj': obj_path, 'npy': numpy_path}
    file_sets = list(file_set.values())
    return file_sets


# Function to split sets into their train-test folders
def train_test_split(root, train_path, test_path, train_ratio=0.8, random_seed=None):
    # Call original function
    sets = split_files_to_sets(root)

    if random_seed is not None:
        np.random.seed(random_seed)

    # Shuffle the dataset
    random.shuffle(sets)

    # Manual function to partition the randomly shuffled set into train and test folders
    partition = int(len(sets) * train_ratio)
    train_sets = sets[:partition]
    test_sets = sets[partition:]

    # Create folders if they do not exist
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # Copy the training set files from the root directory into the training set folder
    for fileset in train_sets:
        for file_type, file_name in fileset.items():
            root_path = os.path.join(root, file_name)
            train_set_path = os.path.join(train_path, file_name)
            shutil.copyfile(root_path, train_set_path)

    # Copy the test set files from the root directory into the test set folder
    for fileset in test_sets:
        for file_type, file_name in fileset.items():
            root_path = os.path.join(root, file_name)
            test_set_path = os.path.join(test_path, file_name)
            shutil.copyfile(root_path, test_set_path)


if __name__ == '__main__':
    train_test_split('../data/', '../data/train_files', '../data/test_files', 0.8)
