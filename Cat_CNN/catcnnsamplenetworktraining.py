
'''This code cell is for making, training, and validating my custom Convolutional network'''
import numpy as np #Importing needed libraries
from tensorflow import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPool2D, Flatten
from keras.preprocessing import image
from keras.regularizers import l1, l2, l1_l2
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
seed = 1
np.random.seed(seed)
random.set_seed(seed)
classifier = Sequential()
classifier.add(Conv2D(64, (3, 3), input_shape = (64, 48, 3), activation = 'relu'))
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 48, 3), activation = 'relu'))
classifier.add(MaxPool2D(2, 2))
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 48, 3), activation = 'relu'))
classifier.add(MaxPool2D(2, 2))
classifier.add(Flatten())
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dropout(rate = 0.1)) 
classifier.add(Dense(64, activation = 'relu'))
classifier.add(Dense(64, activation = 'relu'))
classifier.add(Dense(64, activation = 'relu'))
classifier.add(Dense(64, activation = 'relu'))
classifier.add(Dense(64, activation = 'relu'))
classifier.add(Dense(64, activation = 'relu'))
classifier.add(Dense(1, activation = 'sigmoid'))
classifier.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])
print (classifier.summary())
epochs = 2000
train_datagen = ImageDataGenerator(rescale = 1./255.0, shear_range = 0.3, zoom_range = 0.1, horizontal_flip = True, vertical_flip = True, width_shift_range = 0.2, height_shift_range = 0.2) #Set up image directories
test_datagen = ImageDataGenerator(rescale = 1./255.0, shear_range = 0.3, zoom_range = 0.1, horizontal_flip = True, vertical_flip = True, width_shift_range = 0.2, height_shift_range = 0.2)
train_set = train_datagen.flow_from_directory('/content/Training-70-30', target_size = (64, 48), batch_size = 10, class_mode = 'binary', seed = seed) #Load in images
test_set = test_datagen.flow_from_directory('/content/Testing-70-30', target_size = (64, 48), batch_size = 5, class_mode = 'binary', seed = seed)
history = classifier.fit(train_set, steps_per_epoch = 14, epochs = epochs, validation_data = test_set, validation_steps = 12, shuffle = True, verbose = False)
epochs1 = [i for i in range(epochs)]
plt.scatter(epochs1, history.history['val_accuracy'])
plt.ylabel("Validation Accuracy")
plt.xlabel("Epochs")
plt.show()
plt.scatter(epochs1, history.history["accuracy"])
plt.xlabel("Epochs")
plt.ylabel("Training Accuracy")
plt.show()
