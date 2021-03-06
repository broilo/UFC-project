# -*- coding: utf-8 -*-
"""data_augmentation_deep_lizard

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aJJfTqn-3RYdfeY2cgBc46qTHZz-5yZS
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

TRAIN_SET_PATH = '/content/drive/My Drive/ufc-project/train/'

gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, 
                         height_shift_range=0.1, shear_range=0.15, 
                         zoom_range=0.1, channel_shift_range=10., 
                         horizontal_flip=True)

def get_fighters(file_path=TRAIN_SET_PATH):
  fighters_path = []
  for fighter in os.listdir(file_path):
    fighters_path += [os.path.join(file_path, fighter)]
  return fighters_path

def get_images_from_fighter(fighter_folder_path):
  images_path = []
  for images in os.listdir(fighter_folder_path):
    images_path += [os.path.join(fighter_folder_path, images)]
  return images_path

# Receives the path to one image, generates other images according to gen and save it in a directory
def image_augmentation(image_path, augmentation_folder_path):
  image = np.expand_dims(plt.imread(image_path), 0)
  i = 0
  for batch in gen.flow(image, batch_size=1, save_to_dir=augmentation_folder_path, 
                        save_format='jpeg'):
    i += 1
    if i > 10:
      break

def create_dict(file_path=TRAIN_SET_PATH):
  images_dict = {}
  fighters_list = get_fighters()
  for fighter in fighters_list:
    images_dict[fighter] = get_images_from_fighter(fighter)
  return images_dict

def create_data_augmentation_dataset(file_path=TRAIN_SET_PATH):
  images_dict = create_dict()
  for fighter, images in images_dict.items():
    augmentation_folder = TRAIN_SET_PATH +  os.path.basename(fighter) +'-data-augmentation'
    os.mkdir(augmentation_folder)
    #fighter_name = os.path.dirname(fighter)
    for image in images:
      image_augmentation(image, augmentation_folder)

create_data_augmentation_dataset()