import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np 

model = load_model("C:\\Users\\MSI\\OneDrive\Desktop\\image classification cnn\\image_classfy.keras")
data_cat = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']

img_width = 180
img_height = 180
st.header("Image classification model")
image = st.text_input("Enter image name ","Apple.jpg")

image = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image)
img_bat=tf.expand_dims(img_arr,0)

Predict = model.predict(img_bat)

score = tf.nn.softmax(Predict)
st.image(image)
st.write('Veg/Fruit in image is '+data_cat[np.argmax(score)])
st.write('with accuracy of '+str(np.max(score)*100))

