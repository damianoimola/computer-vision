# from google.colab import files
# !pip install -q kaggle
# # make a diectoryin which kajggle.json is stored
# ! mkdir ~/.kaggle
# ! cp kaggle.json ~/.kaggle/
# ! chmod 600 ~/.kaggle/kaggle.json
# # download the dataset inside colab folder
# !kaggle datasets download -d alessiocorrado99/animals10
# # unzip dataset
# !unzip /content/animals10.zip

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.utils.vis_utils import plot_model
import numpy as np
import os
import cv2

train_data_dir='/content/raw-img'
img_height=128
img_width=128
batch_size=64
nb_epochs=2

def build_model():
    inputShape = (128, 128, 3)
    input = tf.keras.Input(inputShape)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu') (input)
    x = tf.keras.layers.MaxPooling2D((2,2)) (x)
    # x = tf.keras.layers.BatchNormalization() (x)
    # x = tf.keras.layers.Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu') (x)
    x = tf.keras.layers.Dropout(0.4) (x)
    x = tf.keras.layers.Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu') (x)
    x = tf.keras.layers.MaxPooling2D((2,2)) (x)
    # x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.Dropout(0.4) (x)
    x = tf.keras.layers.Flatten() (x)
    output = tf.keras.layers.Dense(10, activation='softmax') (x)
    model = Model(inputs=input, outputs=output)
    plot_model(model, to_file="image-recognition-animal/model.png")
    return model


def predict(model):
    # order of the animals array is important
    # animals=["dog", "horse","elephant", "butterfly",  "chicken",  "cat", "cow",  "sheep","spider", "squirrel"]
    bio_animals = sorted(os.listdir('image-recognition-animal/data/raw-img'))
    categories = {'cane': 'dog',
                  "cavallo": "horse",
                  "elefante": "elephant",
                  "farfalla": "butterfly",
                  "gallina": "chicken",
                  "gatto": "cat",
                  "mucca": "cow",
                  "pecora": "sheep",
                  "scoiattolo": "squirrel",
                  "ragno": "spider"}

    def recognize_class(pred):
        animals = [categories.get(item, item) for item in bio_animals]
        print("The image consist of ", animals[pred])

    img = image.load_img("/content/cane.jpeg", target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    prediction = model.predict(x)
    # prediction

    recognize_class(np.argmax(prediction))

    # test_data_path = "/content/test data/test_animals"
    # files = sorted(os.listdir(test_data_path))
    # files = files[1:]
    # for img in files:
    #     x = cv2.imread(os.path.join(test_data_path, img))
    #     cv2.imshow(x)
    #     recognise(np.argmax(prediction[files.index(img)]))
    #     print("")


def run():
    train_datagen = ImageDataGenerator(rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2) # set validation split

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training') # set as training data

    validation_generator = train_datagen.flow_from_directory(
        train_data_dir, # same directory as training data
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation') # set as validation data

    model = build_model()
    model.summary()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    #train the model,this step takes alot of time (hours)
    model.fit(train_generator,
              steps_per_epoch = train_generator.samples // batch_size,
              validation_data = validation_generator,
              validation_steps = validation_generator.samples // batch_size,
              epochs = nb_epochs)

    #save the model for later use
    model.save('image-recognition-animal/model/model.h')

    predict(model)


def load():
  model = tf.keras.models.load_model('image-recognition-animal/model/model.h')
  predict(model)

