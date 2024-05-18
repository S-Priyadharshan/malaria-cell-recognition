# Deep Neural Network for Malaria Infected Cell Recognition

## AIM

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## Problem Statement and Dataset
The task is to improve the model that identifies malaria-infected cells versus uninfected cells in microscope images. This will help diagnose malaria more accurately. You can enhance the model by adjusting settings, trying different designs, or using methods like transfer learning to boost accuracy.
## Neural Network Model

![image](https://github.com/S-Priyadharshan/malaria-cell-recognition/assets/145854138/db2ce00a-dbf4-4555-8818-03d9f9c8e068)

## DESIGN STEPS

### STEP 1:
Import necessary libraries for data manipulation, visualization, and deep learning.


### STEP 2:
Set up TensorFlow session to dynamically allocate GPU memory and log device placement.


### STEP 3:
Define directory paths for the dataset and inspect their contents.

### STEP 4:
Load sample images for visualization, exploring dimensions, and distributions using seaborn.

### STEP 5:
Define the image shape and construct a sequential model using Keras.

### STEP 6:
Add convolutional, pooling layers, flatten the layer, and add dense layers with activation functions.

### STEP 7:
Compile the model specifying loss function, optimizer, metrics, configure data augmentation, set batch size, train the model, evaluate performance, generate predictions, and perform random image selection for inference with prediction display.

## PROGRAM

### Name: Priyadharshan S

### Register Number: 212223240127

```
Name: Priyadharshan S
Register Number: 212222240127
# to share the GPU
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
config.log_device_placement = True # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from sklearn.metrics import classification_report,confusion_matrix
my_data_dir = './dataset/cell_images'
os.listdir(my_data_dir)
test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'
os.listdir(train_path)
len(os.listdir(train_path+'/uninfected/'))
len(os.listdir(train_path+'/parasitized/'))
os.listdir(train_path+'/parasitized')[200]
para_img= imread(train_path+
                 '/parasitized/'+
                 os.listdir(train_path+'/parasitized')[200])
plt.imshow(para_img)
# Checking the image dimensions
dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'/uninfected'):
    img = imread(test_path+'/uninfected'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)
sns.jointplot(x=dim1,y=dim2)
image_shape = (130,130,3)
help(ImageDataGenerator)
image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )
image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)
model = models.Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(3,3),input_shape=image_shape, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
batch_size = 16
help(image_gen.flow_from_directory)
train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary')
train_image_gen.batch_size
len(train_image_gen.classes)
train_image_gen.total_batches_seen
test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)
train_image_gen.class_indices
results = model.fit(train_image_gen,epochs=3,
                              validation_data=test_image_gen
                             )
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
model.metrics_names
model.evaluate(test_image_gen)
pred_probabilities = model.predict(test_image_gen)
test_image_gen.classes
predictions = pred_probabilities > 0.5
print(classification_report(test_image_gen.classes,predictions))
confusion_matrix(test_image_gen.classes,predictions)
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/S-Priyadharshan/malaria-cell-recognition/assets/145854138/5d295e72-bfed-406a-8721-f3556dd5a2e0)


### Classification Report

![image](https://github.com/S-Priyadharshan/malaria-cell-recognition/assets/145854138/87b5e33a-16e5-48ef-a8b9-a9be4edb5119)


### Confusion Matrix

![image](https://github.com/S-Priyadharshan/malaria-cell-recognition/assets/145854138/fa40e2da-493e-4d30-9081-91e3b3e2ee03)


### New Sample Data Prediction

![image](https://github.com/S-Priyadharshan/malaria-cell-recognition/assets/145854138/af774a6e-9633-4617-ac26-8e01404396a5)


## RESULT
Thus, a deep neural network for Malaria infected cell recognition is developed and the performance is analyzed.
