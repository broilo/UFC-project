import base64
import numpy as np
import io
from flask import request
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
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO
import pickle
import pandas as pd
import json


app = Flask(__name__)

print('* Loading models...')
facenet_model = load_model('./facenet_keras.h5', compile=False)
detector = MTCNN()
model = pickle.load(open("./ufc_2fighterstrained_model.pkl", 'rb'))
out_encoder = pickle.load(open("./ufc_2fighterstrained_out_encoder.pkl", 'rb'))
print("* Loaded")

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
    """Receives a photo represented in a numpy array. Then, makes use of the 
    method detect_faces of the neural network MTCNN. This method returns the 
    position of the bounding box. It, then, returns the numpy array containing 
    only the faces in the image.

    Args:
        image_array (np.ndarray): numpy array representation of the photo.

    Returns:
        np.ndarray: numpy array representation of the photo containing only the
        face found in the photo by the neural network MTCNN.
    """
    faces = detector.detect_faces(image_array)
    all_faces_in_image = []
    i = 0
    for i in range(len(faces)):
        if faces[i]['confidence'] > 0.97:
            x1, y1, width, height = faces[i]['box']
            # Fixing a bug, since sometimes the positions may come as negative numbers
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = image_array[y1:y2, x1:x2]
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

def get_fighters(url):
    face_array_list = extract_face_from_url(url)

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
        
    return predict_names[0], class_probability


@app.route('/predict', methods=['POST'])
def predict():
    """Receives a url photo, process it and make a prediction using SVC model.

    Args:
        url (str): url containing the image.
    """

    # Process the url to get and embedding representation of the face.
    message = request.get_json(force=True)
    
    url = message['name']
    predict_names, class_probability = get_fighters(url)
    prediction = "Predicted: {0} Probability {1:.3f}".format(predict_names, class_probability)
    response = {
        'prediction': prediction
    }

    return jsonify(response)

if __name__ == '__main__':
    
    app.run( host='0.0.0.0', port='5000')