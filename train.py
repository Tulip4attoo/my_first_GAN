import model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model, Model
import train_utils, data_utils
import numpy as np
import os
from tqdm import tqdm
import cv2


NP_PATH = "data/full_numpy_bitmap_camel.npy"
GEN_DAT_PATH = "data/gen_data/"
if not os.path.isdir(GEN_DAT_PATH):
    os.mkdir(GEN_DAT_PATH)

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=1)
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1)
tensorboard = TensorBoard(log_dir='./Graph', histogram_freq=0,
                          write_graph=True, write_images=True)
checkpoint = ModelCheckpoint('ep{epoch:03d}-loss{loss:.3f}.h5',
                             monitor='loss', save_weights_only=True, 
                             save_best_only=True, period=5)

opt_discriminator = RMSprop(lr=0.00002)
opt_gan = RMSprop(lr=0.001)

discriminator = model.discriminator()
discriminator.compile(optimizer=opt_discriminator, loss="binary_crossentropy",
                      callbacks=[reduce_lr, tensorboard, checkpoint])

generator = model.generator()

discriminator.trainable = False
gan_input = Input((100,))
gan_output = discriminator(generator(gan_input))
gan_model = Model(gan_input, gan_output)
gan_model.compile(optimizer=opt_gan, loss="binary_crossentropy",
                  callbacks=[tensorboard, checkpoint])

epochs = 10001
batch_size = 128

x_train = data_utils.load_zeros_mnist()

for epoch in tqdm(range(epochs)):
    discriminator.trainable = True
    train_utils.train_discriminator(discriminator, generator,
                                    x_train, batch_size)
    discriminator.trainable = False
    train_utils.train_generator(gan_model, batch_size)
    if epoch % 500 == 0:
        folder = GEN_DAT_PATH + str(epoch)
        if not os.path.isdir(folder):
            os.mkdir(folder)
            noise = np.random.normal(0, 1, (batch_size, 100))
            gen_imgs = generator.predict(noise)
            for ind in range(len(gen_imgs)):
                img = gen_imgs[ind]
                cv2.imwrite(folder + "/" + str(ind) + ".png", img)

