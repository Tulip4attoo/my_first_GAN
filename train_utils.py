import numpy as np


def train_discriminator(discriminator, generator, x_train,
                        batch_size, z_dim=100):
    """
    """
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    # train on real images
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    true_imgs = x_train[idx]
    discriminator.train_on_batch(true_imgs, valid)

    # train on generated images
    noise = np.random.normal(0, 1, (batch_size, z_dim))
    gen_imgs = generator.predict(noise)
    discriminator.train_on_batch(gen_imgs, fake)
    return discriminator


def train_generator(gan_model, batch_size, z_dim=100):
    """
    """
    valid = np.ones((batch_size, 1))
    noise = np.random.normal(0, 1, (batch_size, z_dim))
    gan_model.train_on_batch(noise, valid)
    return gan_model