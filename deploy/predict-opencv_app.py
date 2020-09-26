import base64
import numpy as np
import io
from flask import request, render_template, make_response, flash
from flask import jsonify
from flask import Flask
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import mtcnn
from PIL import Image
from matplotlib import pyplot
from numpy import asarray
from numpy import load
from numpy import expand_dims
from mtcnn.mtcnn import MTCNN
from typing import Tuple
from typing import List
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO
import pickle
import pandas as pd
import json
from forms import URLRequisition
import cv2
from cv2 import CascadeClassifier


app = Flask(__name__)

app.config['SECRET_KEY'] = 'b53b126b362e1b72adfc3b45044abf33'
#app.config['TEMPLATES_AUTO_RELOAD'] = True

print(' * Loading models...')
facenet_model = load_model('./facenet_keras.h5', compile=False)
classifier = CascadeClassifier('./haarcascade_frontalface_default.xml')
model = pickle.load(open("./ufc-253-trained-model.pkl", 'rb'))
out_encoder = pickle.load(open("./ufc-253-trained-out-encoder.pkl", 'rb'))
print(" * Loaded")

def get_embedding(facenet_model, face_pixels: np.ndarray) -> np.ndarray:
    """Given a face array, it returns a vectorial representation done by the 
    FaceNet neural network.

    Args:
        facenet_model (neural network): neural network that returns an face 
        embedding.
        face_pixels (np.ndarray): face array.

    Returns:
        np.ndarray: emdedding face, a vectorial representation of the face.
    """

    # scale pixel values
    face_pixels = face_pixels.astype('float32')

    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std

    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)

    # make prediction to get embedding
    yhat = facenet_model.predict(samples)

    return yhat[0]


def resize_extracted_face(image_array: np.ndarray, 
                          required_size: Tuple[int, int] = (160, 160)) -> np.ndarray:
    """Receives the a numpy array representation of the image, and returns it 
    reshaped by the requireded size.

    Args:
        image_array (np.ndarray): numpy array representation of the photo.
        required_size (Tuple[int, int], optional): Image size. Defaults to 
        (160, 160).

    Returns:
        np.ndarray: Resized image.
    """
    image = Image.fromarray(image_array).resize(required_size)
    face_array = asarray(image)

    return face_array


def normalize_one_array(image_array: np.ndarray) -> np.ndarray:
    """Receives image array and returns it normalized.

    Args:
        image_array (np.ndarray): array representation of image.

    Returns:
        np.ndarray: normalized array.
    """
    in_encoder = Normalizer(norm='l2')
    normalized_array = in_encoder.transform(image_array)

    return normalized_array

def get_image_array_from_url(url: str) -> np.ndarray:
    """Receives an url and returns the image array.

    Args:
        url (str): url containing the image.

    Returns:
        np.ndarray: numpy array representation of the image.
    """

    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    pixels = np.asarray(img)
        
    return pixels


def extract_all_faces_from_image_array(image_array: np.ndarray) -> np.ndarray:
    
    faces = classifier.detectMultiScale(image_array, minNeighbors=8)
    all_faces_in_image = []
    
    for box in faces:
        x, y, width, height = box
        face = image_array[y:y+width, x:x+height]
        all_faces_in_image.append(face)

    return all_faces_in_image


def extract_face_from_url(url: str) -> np.ndarray:
    """Given an url, it gets the image array with the method 
    get_image_array_from_url and then makes use o the neural network MCTNN to 
    get the face array. It returns the resized array.

    Args:
        url (str): url containing the image.

    Returns:
        np.ndarray: numpy array representation of the face.
    """
    image_array = get_image_array_from_url(url)
    face_array_list = extract_all_faces_from_image_array(image_array)
    face_array_list_resized = []
    for face_array in face_array_list:
        # resizing each face
        resized_face_array = resize_extracted_face(face_array)
        face_array_list_resized.append(resized_face_array)

    return face_array_list_resized


def get_information(name: str):
    """Receives a fighter name, and then make a request to get information about
    it in ufc repository.
     Args:
        name (str): fighters name.
    """
    df = pd.read_csv("./fighters_info/fighters_info.csv")
    #name = name.split('-')
    #name = name[0] + ' ' + name[1]
    
    return df[df['Nome'] == name]

def past_fights(name: str):
    path = f"./fighters_info/{name}.csv"
    df = pd.read_csv(path)

    return df

def get_fighters(url):
    face_array_list = extract_face_from_url(url)
    fighters_list = []
    for face_array in face_array_list:
        face_embedding = get_embedding(facenet_model, face_array)
        face_embedding = expand_dims(face_embedding, axis=0)
        face_embedding_normalized = normalize_one_array(face_embedding)

        # Make a prediction
        yhat_class = model.predict(face_embedding_normalized)
        yhat_prob = model.predict_proba(face_embedding_normalized)

        # Get names and probabilities
        class_index = yhat_class[0]
        class_probability = yhat_prob[0, class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)

        if class_probability < 94:
            predict_names[0] = 'Unknown'
            

        #dictionary to store fighter info    
        fighter_dict = {}
        fighter_dict['name'] = predict_names[0]
        fighter_dict['probability'] = str(class_probability)

        if predict_names[0] == 'Unknown':
            fighter_dict['info'] = 'No info'
            fighter_dict['past_fights'] = 'No fights'
        else:
            df = get_information(predict_names[0])
            df = df.set_index('Nome')
            df.drop(['Unnamed: 0'], axis =1, inplace=True)
            fighter_dict['info'] = df.to_html()
            fighter_dict['past_fights'] = past_fights(predict_names[0]).to_html()

        plt.imshow(face_array)
        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        figfile.seek(0)  # rewind to beginning of file

    #figdata_png = base64.b64encode(figfile.read())
        figdata_png = base64.b64encode(figfile.getvalue())
        
        fighter_dict['photo'] = figdata_png

        #fighter_dict['past_fights'] = past_fights(predict_names[0])


        #appending this dict to a list, to store info about all fighters
        fighters_list.append(fighter_dict)
        
    return fighters_list


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Receives a url photo, process it and make a prediction using SVC model.

    Args:
        url (str): url containing the image.
    """
    form = URLRequisition()
    
    # Process the url to get and embedding representation of the face.
    if form.validate_on_submit():
        
        url = form.url.data
        print(url)
        posts = get_fighters(url)
        form = URLRequisition()
        return render_template('predict.html', form=form, posts=posts)

    return render_template('predict.html', form=form)

if __name__ == '__main__':
    
    app.run( host='0.0.0.0', port='8080')
    #app.run( host='0.0.0.0', port='5000')