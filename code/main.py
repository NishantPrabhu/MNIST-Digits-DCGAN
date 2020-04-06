# ============================================================= #
# Forecasting Stock Prices with GANs and RL                     # 
# The Analytics Club, Center for Innovation, IIT Madras         #
# ------------------------------------------------------------- #
#                                                               #
# Deep Convolutional Generative Adversarial Network             #
# Implementation by Nishant Prabhu                              #
#                                                               #
# Generate new MNIST handwritten digits with this barebones     #
# implementation of DCGAN                                       #    
# ============================================================= #

# Dependencies
import numpy as np 
import tensorflow as tf
from tensorflow.keras.datasets import mnist 
from dcgan import DCGAN 


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config = config)

def main():
    
    # Load MNIST data
    print("[INFO] Loading MNIST handwritten digits...\n")
    (X, _), (_, _) = mnist.load_data()
    images = X.reshape(X.shape[0], 28, 28, 1)

    # Initialize DCGAN object
    gan_model = DCGAN(
        data = images,
        learning_rate = 2e-04,
        batch_size = 128,
        latent_dim = 100
    )

    # Train GAN Model
    gan_model.train(
        epochs = 100, 
        batches_per_epoch = 300, 
        checkpoint_frequency = 1,
        save_path = '.././save_data/'    
    )


if __name__ == "__main__":
    
    # calling the main function defined above.
    main()
