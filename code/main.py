from DataLoader import DataLoader, train_test_split
from train import train_vae
from test import test_vae


# Parameters
img_size = 28

# Run training
if __name__ == '__main__':
    train_test_split("../data/", "../data/train_files", "../data/test_files")
    training_data = DataLoader("../data/train_files")
    test_data = DataLoader("../data/test_files")
    train_vae(training_data, epochs=2, batch_size=2)
    test_vae(test_data, batch_size=2)
