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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten
from tensorflow.keras.layers import LeakyReLU, Dropout, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
import time
from tqdm import tqdm


# Class DCGAN
class DCGAN():

    def __init__(self, data, learning_rate, batch_size, latent_dim):
        self.data = data
        self.lr = learning_rate
        self.batch_size = batch_size
        self.latent_dim = latent_dim

        # Create discriminator and generator
        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()
        
        # Initialize GAN and discriminator
        self.GAN = Sequential()
        self.GAN.add(self.generator)
        self.GAN.add(self.discriminator)
        
        # Compile GAN and discriminator
        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=10*self.lr), metrics=['accuracy'])
        self.GAN.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.lr))
        
        # Pretrain discriminator
        self.pretrain_discriminator(n_samples=1000, epochs=3)

    def create_generator(self):
        model = Sequential()
        model.add(Dense(128*7*7, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((7, 7, 128)))
        model.add(Conv2DTranspose(64, kernel_size=(4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(128, kernel_size=(4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(1, kernel_size=(5,5), padding='same', activation='tanh'))
        return model

    def create_discriminator(self):
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3,3), strides=(2,2), padding='same', input_shape=(28, 28, 1)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Conv2D(64, kernel_size=(3,3), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model

    def generate_latent_samples(self, size, invert_labels=False):
        noise = np.random.randn(size * self.latent_dim).reshape((size, self.latent_dim))
        if invert_labels is True:
            labels = np.asarray([0]*size)
            return noise, labels
        return noise

    def generate_batch(self, size):
        rand_idx = np.random.randint(0, len(self.data), size=size//2)
        real_ims = self.data[rand_idx]
        real_ims = real_ims.reshape(real_ims.shape[0], 28, 28, 1)
        noise = self.generate_latent_samples(size=size//2, invert_labels=False)
        fake_ims = self.generator.predict(noise)
        labels = np.asarray([0]*(size//2) + [1]*(size//2))
        ims = np.vstack((real_ims, fake_ims))
        return ims, labels

    def pretrain_discriminator(self, n_samples=1000, epochs=3):
        x, y = self.generate_batch(n_samples)
        print('\n')
        self.discriminator.fit(x, y, epochs=epochs)
        time.sleep(3)

    def test_generator(self, n_samples=5):
        noise = self.generate_latent_samples(size=n_samples, invert_labels=False)
        outs = self.generator.predict(noise)
        for i in range(outs.shape[0]):
            plt.add_subplot(1, n_samples, i+1)
            plt.axis('off')
            plt.imshow(outs[i, :, :, 0].reshape(28, 28), cmap='gray')
        plt.show()
        print('\n')

    def discriminator_trainable(self, val):
        self.discriminator.trainable = val
        for l in self.discriminator.layers:
            l.trainable = val

    def train(self, epochs, batches_per_epoch, checkpoint_frequency, save_path):
        for i in range(epochs):
            print("Epoch {}/{}".format(i+1, epochs))
            
            for j in tqdm(range(batches_per_epoch), position=0, leave=False):
                
                # Generate training batch
                x, y = self.generate_batch(size=self.batch_size)
                
                # Train discriminator
                self.discriminator.train_on_batch(x, y)
                
                # Render discriminator untrainable
                self.discriminator_trainable(False)
                
                # Generate batch for GAN
                x_gan, y_gan = self.generate_latent_samples(size=self.batch_size, invert_labels=True)
                
                # Train GAN
                self.GAN.train_on_batch(x_gan, y_gan)
                
                # Render discriminator trainable
                self.discriminator_trainable(True)

            if i % checkpoint_frequency == 0:
                self.test_generator(n_samples=5)
                self.generator.save(save_path+'generator_'+str(i)+'.h5')
                self.GAN.save(save_path+'GAN_'+str(i)+'.h5')