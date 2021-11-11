import matplotlib
matplotlib.use('Agg')

import os
from keras import models
import requests
from flask import Flask, request, jsonify

import librosa
from librosa.core import convert
import librosa.display

import tensorflow as tf
from tensorflow import keras
from keras import models
import keras.backend as K

import numpy as np

from keras import layers
from keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, 
                          Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,
                          Dropout)
from keras.models import Model, load_model

from keras.initializers import glorot_uniform
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img,img_to_array

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import streamlit as st
from streamlit import caching

from PIL import Image
import librosa
import numpy as np
import librosa.display
from pydub import AudioSegment
import matplotlib.cm as cm
from matplotlib.colors import Normalize
  
app = Flask(__name__)
model = None
SAVED_MODEL_PATH = "model.h5"

def _load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = load_model("./model.h5")
    
def convert_mp3_to_wav(music_file):
    sound = AudioSegment.from_file(music_file,format = "mp3")
    sound.export("./music_file.wav",format="wav")

def extract_relevant(wav_file,t1,t2):
    wav = AudioSegment.from_wav(wav_file)
    wav = wav[1000*t1:1000*t2]
    wav.export("./extracted.wav",format='wav')
        
def create_melspectrogram(wav_file):
    y,sr = librosa.load(wav_file,duration=3)
    mels = librosa.feature.melspectrogram(y=y,sr=sr)

    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    p = plt.imshow(librosa.power_to_db(mels,ref=np.max))
    plt.savefig('./melspectrogram.png')

@app.route("/predict", methods=["POST"])
def predict():
    
    audio_file = request.files["file"]
    audio_file.save("./song.mp3")
    
    #load_model
    _load_model()
    
    #preprocessing
    convert_mp3_to_wav("./song.mp3")    
    extract_relevant("./music_file.wav",40,50)
    create_melspectrogram("./extracted.wav") 
    image_data = load_img('./melspectrogram.png',color_mode='rgba',target_size=(288,432))
    
    
    #predict
    image = img_to_array(image_data)
    image = np.reshape(image,(1,288,432,4))
    prediction = model.predict(image/255)
    prediction = prediction.reshape((6,)) 
    class_label = np.argmax(prediction)
    
    class_labels = ['rythmandblues','country', 'disco' ,'hiphop', 'jazz', 'rock']
    
    result = {"genre": class_labels[class_label]}
    print(result)
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=False)