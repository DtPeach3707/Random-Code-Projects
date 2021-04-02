
"""Card CGAN

Code supposed to be executed through Google Colab Notebook. Original found here: https://colab.research.google.com/drive/1_XUcRF-DqZS-osdDkwXEX4MGdcgzYwGs?usp=sharing
(Reformatted so it could all be executed in one cell)

zip folder and google drive dataset made by Maxim Ziatdinov (citation below)
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import concatenate
from keras.optimizers import RMSprop
from keras.models import Model
from keras.utils import to_categorical
!gdown https://drive.google.com/uc?id=1AyGHVflbIjzinkKBURHNVDx1wWg9JixB
!unzip cards.zip
card1 = cv2.resize(cv2.imread("cards/card1.JPG", cv2.IMREAD_GRAYSCALE), (48, 48))
card2 = cv2.resize(cv2.imread("card/card2.JPG", cv2.IMREAD_GRAYSCALE), (48, 48))
card3 = cv2.resize(cv2.imread("cards/card3.JPG", cv2.IMREAD_GRAYSCALE), (48, 48))
card4 = cv2.resize(cv2.imread("cards/card4.JPG", cv2.IMREAD_GRAYSCALE), (48, 48))
cv2.imwrite('/content/Card_1.jpg', card1)
cv2.imwrite('/content/Card_2.jpg', card2)
cv2.imwrite('/content/Card_3.jpg', card3)
cv2.imwrite('/content/Card_4.jpg', card4)
card1 = Image.open('/content/Card_1.jpg')
card2 = Image.open('/content/Card_2.jpg')
card3 = Image.open('/content/Card_3.jpg')
card4 = Image.open('/content/Card_4.jpg')
def leftshift(image, n):
  image = np.array(image)
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      if (i < image.shape[1] - n):
        image[j][i] = image[j][i + n]
  return image
def rightshift(image, n):
  image = np.array(image)
  for i in range(image.shape[0], 1, -1):
    for j in range(image.shape[1]):
      if (i < image.shape[0] - n):
        image[j][i] = image[j][i - n]
  return image
def upshift(image, n):
  image = np.array(image)
  for j in range(image.shape[0]):
    for i in range(image.shape[1]):
      if (j < image.shape[0] - n and j > n):
        image[j][i] = image[j + n][i]
  return image
def downshift(image, n):
  image = np.array(image)
  for j in range(image.shape[0], 1, -1):
    for i in range(image.shape[1]):
      if (j > n and j < image.shape[0] - n):
        image[j][i] = image[j - n][i]
  return image
plt.imshow(downshift(card1, 6))
def transform_preprocess(image):
  final_images = []
  horzflip = image.transpose(method = Image.FLIP_LEFT_RIGHT)
  vertflip = image.transpose(method = Image.FLIP_TOP_BOTTOM)
  reflflip = horzflip.transpose(method = Image.FLIP_TOP_BOTTOM)
  images = [image, horzflip, vertflip, reflflip]
  for image in [image, horzflip]:
    images.append(image.rotate(90))
    images.append(image.rotate(270))
  for image in images:
    for m in range(9):
      final_images.append(leftshift(image, m))
      for l in range(6):
        final_images.append(upshift(leftshift(image, m), l))
        final_images.append(downshift(leftshift(image, m), l))
    for m in range(6):
      final_images.append(rightshift(image, m))
      for l in range(6):
        final_images.append(upshift(rightshift(image, m), l))
        final_images.append(downshift(rightshift(image, m), l))
    final_images.append(np.array(image))
  final_images = np.array(final_images)
  return final_images
index_array = []
Card1 = transform_preprocess(card1)
for i in range(Card1.shape[0]):
  index_array.append(0)
Card2 = transform_preprocess(card2)
for i in range(Card2.shape[0]):
  index_array.append(1)
Card3 = transform_preprocess(card3)
for i in range(Card3.shape[0]):
  index_array.append(2)
Card4 = transform_preprocess(card4)
for i in range(Card4.shape[0]):
  index_array.append(3)
index_array = np.array(index_array)
FinalCards = np.concatenate((Card1, Card2, Card3, Card4), axis = 0)
def build_generator(inputs, labels, image_size):
    image_resize = (image_size[0] // 4, image_size[1] // 4)
    kernel_size = 5
    layer_filters = [128, 64, 32, 1]
    x = concatenate([inputs, labels], axis=1)
    x = Dense(image_resize[0] * image_resize[1] * layer_filters[0])(x)
    x = Reshape((image_resize[0], image_resize[1], layer_filters[0]))(x)
    for filters in layer_filters:
        if filters > layer_filters[-2]:
            strides = 2
        else:
            strides = 1
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = Reshape((48, 48, 1))(x)
    x = Activation('sigmoid')(x)
    generator = Model([inputs, labels], x, name='generator')
    return generator
def build_discriminator(inputs, labels, image_size):
    kernel_size = 5
    layer_filters = [32, 64, 128, 256]
    x = inputs
    y = Dense(image_size[0] * image_size[1])(labels)
    y = Reshape((image_size[0], image_size[1], 1))(y)
    x = concatenate([x, y])
    for filters in layer_filters:
        if filters == layer_filters[-1]:
            strides = 1
        else:
            strides = 2
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same')(x)
    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    discriminator = Model([inputs, labels], x, name='discriminator')
    return discriminator
def train(models, data, params):
    losss = []
    accc = []
    generator, discriminator, adversarial = models
    x_train, y_train = data
    batch_size, latent_size, train_steps, num_labels, model_name = params
    save_interval = 500
    noise_input = np.random.uniform(-1.0, 1.0, size=[64, latent_size])
    noise_class = np.eye(num_labels)[np.arange(0, 64) % num_labels]
    print (noise_class)
    train_size = x_train.shape[0]
    print(model_name, "Labels for generated images: ", np.argmax(noise_class, axis=1))
    accavg = 0
    accavg1 = 0
    epochs = [i for i in range(train_steps)]
    frames = []
    for i in range(train_steps):
        rand_indexes = np.random.randint(0, train_size, size=batch_size)
        real_images = x_train[rand_indexes]
        real_labels = y_train[rand_indexes]
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels, batch_size)]
        fake_images = generator.predict([noise, fake_labels])
        x = np.concatenate((real_images, fake_images))
        labels = np.concatenate((real_labels, fake_labels))
        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0.0
        loss, acc = discriminator.train_on_batch([x, labels], y)
        losss.append(loss)
        accc.append(acc)
        log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)
        accavg += acc
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels, batch_size)]
        y = np.ones([batch_size, 1])
        loss, acc = adversarial.train_on_batch([noise, fake_labels], y)
        log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)
        accavg1 += acc
        print(log)
        if (i + 1) % save_interval == 0:
            accavg = accavg / save_interval
            accavg1 = accavg1 / save_interval
            print ("Average discriminator accuracy: " + str(accavg))
            print ("Average adversarial accuracy: " + str(accavg1))
            accavg = 0
            accavg1 = 0
            plot_images(generator, noise_input, noise_class, show = True, step = i + 1)
            images = generator.predict([noise_input, noise_class])
            frames.append(np.array(images[0]).reshape((48, 48)))
    plt.plot(epochs, losss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss')
    plt.show()
    plt.plot(epochs, accc)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Discriminator Accuracy")
    plt.show()
    generator.save(model_name + ".h5")
def plot_images(generator, noise_input, noise_class, show=False, step=0, model_name="gan"):
    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, "%05d.png" % step)
    images = generator.predict([noise_input, noise_class])
    print(model_name , " labels for generated images: ", np.argmax(noise_class, axis=1))
    plt.figure(figsize=(11.1, 11.1))
    num_images = images.shape[0]
    image_size = images.shape[1]
    rows = int(math.sqrt(noise_input.shape[0]))
    for i in range(num_images):
        plt.subplot(rows, rows, i + 1)
        plt.imshow(np.array(images[i - 1]).reshape((48, 48)))
        plt.axis('off')
    plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close('all')
def build_and_train_models():
    x_train = FinalCards
    y_train = index_array
    y_train = to_categorical(y_train)
    x_train = np.reshape(x_train, [x_train.shape[0], x_train.shape[1], x_train.shape[2], 1])
    x_train = x_train.astype('float32') / 255
    model_name = "cgan_card"
    latent_size = 2
    batch_size = 64
    train_steps = 40000
    lr = 1e-4
    decay = 6e-8
    input_shape = (48, 48, 1)
    label_shape = (4,)
    image_size = (48, 48)
    inputs = Input(shape=input_shape, name='discriminator_input')
    labels = Input(shape=label_shape, name='class_labels')
    discriminator = build_discriminator(inputs, labels, image_size)
    optimizer = RMSprop(lr=lr, decay=decay)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    discriminator.summary()
    input_shape = (latent_size, )
    inputs = Input(shape=input_shape, name='z_input')
    generator = build_generator(inputs, labels, image_size)
    generator.summary()
    optimizer = RMSprop(lr=lr*0.75, decay=decay*0.75)
    discriminator.trainable = False
    outputs = discriminator([generator([inputs, labels]), labels])
    adversarial = Model([inputs, labels], outputs, name=model_name)
    adversarial.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    adversarial.summary()
    models = (generator, discriminator, adversarial)
    data = (x_train, y_train)
    num_labels = 4
    params = (batch_size, latent_size, train_steps, num_labels, model_name)
    train(models, data, params)
build_and_train_models()
'''
Code for making the gif
'''
datadir = '/content/gan'
filelist = sorted(os.listdir(datadir))
frames = []
for fil in filelist:
  path = '/content/gan/' + fil
  fil = Image.open(path)
  frames.append(fil)
frames[0].save('Card_Training.gif', format='GIF', append_images=frames[1:], save_all=True, duration = 300, loop = 0)

'''
1. Ziatdinov, M.; Kalinin, S. V., Enter the j(r)VAE: divide, (rotate), and order... the cards. 
Towards data Science. 2021. https://towardsdatascience.com/enter-the-j-r-vae-divide-rotate-and-order-the-cards-9d10c6633726
'''
