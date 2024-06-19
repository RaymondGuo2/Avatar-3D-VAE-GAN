from DataLoader import DataLoader
from train import train_vae


# Parameters
img_size = 28

# Run training
if __name__ == '__main__':
    path_data = "../data"
    dataset = DataLoader(path_data, image_size=img_size)
    train_vae(dataset, epochs=2, batch_size=2)
