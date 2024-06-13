This is a replication of the 3D-VAE-GAN in the paper "Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling" by Wu et al. (2016).


To facilitate the convolutional networks, the original mesh representations must be converted into voxels. A popular approach is Patrick Min's binvox package found here: https://www.patrickmin.com/binvox/ . To further facilitate the model, the binvox-rw-py package is used to convert the binary occupancy grid format into a 3D numpy array, created by Daniel Maturana (2012).