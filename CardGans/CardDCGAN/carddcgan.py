"""
Code supposed to be executed through Google Colab

Original notebook where the code comes from can be found here: https://colab.research.google.com/drive/1RZ6x-GNPQfbL1eWMYPU0S6g-UAZybUAQ?usp=sharing
(slightly reformatted so it can all run in one cell)

Card data credited to Maxim Ziatdinov (citation below)

Has all the code to generate the gif, the classifier, the dcgan h5, and the graph
"""

!gdown https://drive.google.com/uc?id=1AyGHVflbIjzinkKBURHNVDx1wWg9JixB
!unzip cards.zip
'''
Importing libraries
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import pdb
import os
from PIL import Image
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import concatenate
from keras.optimizers import RMSprop
from keras.models import Model
from keras.utils import to_categorical
from keras.models import load_model
'''
Preprocessing images with resize and image augmentation
'''
card1 = cv2.resize(cv2.imread("cards/card1.JPG", cv2.IMREAD_GRAYSCALE), (48, 48))
card2 = cv2.resize(cv2.imread("cards/card2.JPG", cv2.IMREAD_GRAYSCALE), (48, 48))
card3 = cv2.resize(cv2.imread("cards/card3.JPG", cv2.IMREAD_GRAYSCALE), (48, 48))
card4 = cv2.resize(cv2.imread("cards/card4.JPG", cv2.IMREAD_GRAYSCALE), (48, 48))
plt.imshow(card1)
plt.show()
plt.imshow(card2)
plt.show()
plt.imshow(card3)
plt.show()
plt.imshow(card4)
plt.show()
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
index_array1 = []
index_array2 = []
index_array3 = []
index_array4 = []
Card1 = transform_preprocess(card1)
for i in range(Card1.shape[0]):
  index_array1.append(0)
Card2 = transform_preprocess(card2)
for i in range(Card2.shape[0]):
  index_array2.append(0)
Card3 = transform_preprocess(card3)
for i in range(Card3.shape[0]):
  index_array3.append(0)
Card4 = transform_preprocess(card4)
for i in range(Card4.shape[0]):
  index_array4.append(0)
FinalCards = np.concatenate((Card1, Card2, Card3, Card4), axis = 0)
'''
Code for building, training, and saving the classifier
'''
import numpy as np #Importing needed libraries
from keras import losses
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPool2D, Flatten
from sklearn.model_selection import train_test_split
Xtrain_set1, Xtest_set1, ytrain1, ytest1 = train_test_split(Card1, index_array1, test_size = 0.25)
Xtrain_set2, Xtest_set2, ytrain2, ytest2 = train_test_split(Card2, index_array2, test_size = 0.25)
Xtrain_set3, Xtest_set3, ytrain3, ytest3 = train_test_split(Card3, index_array3, test_size = 0.25)
Xtrain_set4, Xtest_set4, ytrain4, ytest4 = train_test_split(Card4, index_array4, test_size = 0.25)
X_train = np.concatenate((Xtrain_set1, Xtrain_set2, Xtrain_set3, Xtrain_set4)) 
X_test = np.concatenate((Xtest_set1, Xtest_set2, Xtest_set3, Xtest_set4))
y_train = np.concatenate((ytrain1, ytrain2, ytrain3, ytrain4))
y_test = np.concatenate((ytest1, ytest2, ytest3, ytest4))
model = Sequential([Flatten(input_shape=(48, 48)), Dense(128, activation='relu'), Dense(4)])
model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(X_train, y_train, epochs = 40)
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
print('\nTest accuracy:', test_acc)
model.save('Card_Classifier.h5')
'''
Code for building, training, and saving the dcGAN
'''
def build_generator(inputs, image_size):
    image_resize = (image_size[0] // 4, image_size[1] // 4)
    kernel_size = 5
    layer_filters = [128, 64, 32, 1]
    x = inputs
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
    generator = Model(inputs, x, name='generator')
    return generator
def build_discriminator(inputs, image_size):
    kernel_size = 5
    layer_filters = [32, 64, 128, 256]
    x = inputs
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
    discriminator = Model(inputs, x, name='discriminator')
    return discriminator
def train(models, data, params):
    losss = []
    accc = []
    generator, discriminator, adversarial = models
    x_train = data
    batch_size, latent_size, train_steps, model_name = params
    save_interval = 500
    noise_input = np.random.uniform(-1.0, 1.0, size=[64, latent_size])
    train_size = x_train.shape[0]
    accavg = 0
    accavg1 = 0
    epochs = [i for i in range(train_steps)]
    frames = []
    for i in range(train_steps):
        rand_indexes = np.random.randint(0, train_size, size=batch_size)
        real_images = x_train[rand_indexes]
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        fake_images = generator.predict(noise)
        x = np.concatenate((real_images, fake_images))
        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0.0
        loss, acc = discriminator.train_on_batch(x, y)
        losss.append(loss)
        accc.append(acc)
        log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)
        accavg += acc
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        y = np.ones([batch_size, 1])
        loss, acc = adversarial.train_on_batch(noise, y)
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
            plot_images(generator, noise_input, show = True, step = i + 1)
            images = generator.predict(noise_input)
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
def plot_images(generator, noise_input, show=False, step=0, model_name="gan"):
    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, "%05d.png" % step)
    images = generator.predict(noise_input)
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
    x_train = np.reshape(x_train, [x_train.shape[0], x_train.shape[1], x_train.shape[2], 1])
    x_train = x_train.astype('float32') / 255
    model_name = "dcgan_card"
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
    discriminator = build_discriminator(inputs, image_size)
    optimizer = RMSprop(lr=lr, decay=decay)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    discriminator.summary()
    input_shape = (latent_size, )
    inputs = Input(shape=input_shape, name='z_input')
    generator = build_generator(inputs, image_size)
    generator.summary()
    optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)
    discriminator.trainable = False
    outputs = discriminator(generator(inputs))
    adversarial = Model(inputs, outputs, name=model_name)
    adversarial.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    adversarial.summary()
    models = (generator, discriminator, adversarial)
    data = x_train
    params = (batch_size, latent_size, train_steps, model_name)
    train(models, data, params)
def test_generator(generator, class_label=None):
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
    step = 0
    if class_label is None:
        num_labels = 10
        noise_class = np.eye(num_labels)[np.random.choice(num_labels, 16)]
    else:
        noise_class = np.zeros((16, 10))
        noise_class[:,class_label] = 1
        step = class_label
    plot_images(generator, noise_input=noise_input, noise_class=noise_class, show=True, step=step, model_name="test_outputs")
build_and_train_models()
'''
Code to generate gif
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
Code to plot latent space graph
Assumes you have classifier loaded into the runtime
'''
generator = load_model('/content/dcgan_card.h5')
classifier = load_model('/content/Card_Classifier.h5')
latent_vals = []
for i in range(500):
  for j in range(500):
    array = np.array((i/500, j/500))
    latent_vals.append(array)
latent_vals = np.array(latent_vals)
images = generator.predict(latent_vals)
classes = classifier.predict(images)
class_list = []
for prediction in classes:
  prediction = list(prediction)
  max_value = max(prediction)
  max_index = prediction.index(max_value)  
  class_list.append(max_index)
k = 0
clubslis = []
spadeslis = []
heartslis = []
diamondslis = []
for val in latent_vals:
  if class_list[k] == 0:
    clubslis.append(val)
  elif class_list[k] == 1:
    spadeslis.append(val)
  elif class_list[k] == 2:
    heartslis.append(val)
  else:
    diamondslis.append(val)
  k += 1
clubsx = []
clubsy = []
spadesx = []
spadesy = []
heartsx = []
heartsy = []
diamondsx = []
diamondsy = []
for x, y in clubslis:
  clubsx.append(x)
  clubsy.append(y)
for x, y in spadeslis:
  spadesx.append(x)
  spadesy.append(y)
for x, y in heartslis:
  heartsx.append(x)
  heartsy.append(y)
for x, y in diamondslis:
  diamondsx.append(x)
  diamondsy.append(y)
plt.scatter(clubsx, clubsy, label = 'clubs')
plt.scatter(spadesx, spadesy, label = 'spades')
plt.scatter(heartsx, heartsy, label = 'hearts')
plt.scatter(diamondsx, diamondsy, label = 'diamond')
plt.xlabel("l1")
plt.ylabel('l2')
plt.title("Latent space values' influence on class")
plt.legend(loc = 'upper left')
plt.savefig('latent_space_graph.png')
plt.show()
'''
1. Ziatdinov, M.; Kalinin, S. V., Enter the j(r)VAE: divide, (rotate), and order... the cards. 
Towards data Science. 2021. https://towardsdatascience.com/enter-the-j-r-vae-divide-rotate-and-order-the-cards-9d10c6633726
'''
