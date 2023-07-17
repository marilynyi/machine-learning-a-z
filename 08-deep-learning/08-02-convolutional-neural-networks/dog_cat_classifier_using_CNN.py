"""
###################################
08.02. Dog/Cat Classifier using CNN
###################################
"""

# Importing the libraries
from IPython.display import Image, display
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

# -------------------------------------------------------------------------------#
# Part 1 - Data Preprocessing
# -------------------------------------------------------------------------------#

# Toggle between abridged and full dataset
use_full_set = True
if use_full_set == True:
    data_path = "config/dataset"
else:
    data_path = "dataset"
    
# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory(f'{data_path}/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(f'{data_path}/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# -------------------------------------------------------------------------------#
# Part 2 - Building the CNN
# -------------------------------------------------------------------------------#

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# -------------------------------------------------------------------------------#
# Part 3 - Training the CNN
# -------------------------------------------------------------------------------#

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

# -------------------------------------------------------------------------------#
# Part 4 - Making a single prediction
# -------------------------------------------------------------------------------#

# Change number between 1-10 to test different predictions
img_num = 10
img_path = f'{data_path}/single_prediction/cat_or_dog_{img_num}.jpeg'
test_image = image.load_img(img_path, target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image/255.0)
training_set.class_indices
if result[0][0] > 0.5:
  prediction = 'dog'
else:
  prediction = 'cat'
  
print(prediction)
display(Image(filename=img_path))

