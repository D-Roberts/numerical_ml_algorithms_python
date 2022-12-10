"""
Classify cifar 10
basic with efficientnet - the 1HR ML model series
"""
import tensorflow as tf 
import numpy as np 

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers.experimental import preprocessing

import pandas as pd

import os 

SIZE = (32, 32)

class DataProcessor:
    def __init__(self):
        pass 

    def load(self):
        image_path = [os.path.join("cifar10/train", fname) for fname in os.listdir("cifar10/train")
        if fname.endswith(".png")]
        # print(image_path)

        X = np.zeros((len(image_path), 32, 32, 3), dtype="float32")
        for i, p in enumerate(image_path):
            im = load_img(p, target_size = (32, 32))
            X[i] = im
        # print(X[0])
        labels = [y for x, y in pd.read_csv("cifar10/trainLabels.csv").values]
     
        labels_set = list(set(labels))
        print(labels_set)
        labels_set.sort()
        label2ind = {x:i for i, x in enumerate(labels_set)}
        newlabels = np.array([label2ind[x] for x in labels], dtype="int32")
        
        dataset = tf.data.Dataset.from_tensor_slices(((tf.constant(X)), tf.constant(newlabels)))
        
        def to_hot(im, lab):
            return im, tf.one_hot(lab, 10)

        dataset = dataset.map(to_hot)
        
        def resize_im(im, lab):
            return tf.image.resize(im, (224, 224)), lab

        dataset = dataset.map(resize_im)
        return dataset.batch(64)

from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0

# img_rescale = Sequential([
#     preprocessing.Rescaling(1./255)
# ])

img_aug = Sequential([
    preprocessing.RandomRotation(factor=0.15),
    preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
    preprocessing.RandomContrast(factor=0.1),
    preprocessing.RandomFlip()
])
def build_net():
    inputs = tf.keras.layers.Input((224, 224, 3))

    x = img_aug(inputs)

    #scale after augment; within model
    x = preprocessing.Rescaling(1./255)(x)

    model = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=x)
    model.trainable = False 

    x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics = ["accuracy"])

    return model




def run_train_eval():
    dp = DataProcessor()
    dp.load()
    train = dp.load()

    model = build_net()

    model.fit(train, epochs=4)

run_train_eval()
