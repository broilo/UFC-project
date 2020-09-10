# pip install mtcnn

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import mtcnn
from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from numpy import load
from numpy import expand_dims
from mtcnn.mtcnn import MTCNN
from typing import Tuple
from typing import List
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from random import choice
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
import requests
from io import BytesIO
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pandas as pd

TRAIN_SET_PATH = './train'
TEST_SET_PATH = './test'
PATH_TO_LOAD_FACENET_MODEL = './facenet_keras.h5'
UFC_PROJECT_PATH = './'
SAVE_NPZ = './'
TRAINED_MODEL = "ufc-253-trained-model.pkl"
TRAINED_OUT_ENCODER = "ufc-253-trained-out-encoder.pkl"
URL_TEST1 = 'https://cdn.vox-cdn.com/thumbor/jKijtSQjaN-w_MqFL5-QUpTpdFs=/0x0:3586x2476/1200x800/filters:focal(880x179:1452x751)/cdn.vox-cdn.com/uploads/chorus_image/image/67132969/993547388.jpg.0.jpg'
URL_TEST2 = 'https://i.superesportes.com.br/PRP6_R8xdJF0WIXKkVZzhMDMG-I=/smart/imgsapp.mg.superesportes.com.br/app/noticia_126420360808/2019/10/06/2532165/20191006022136907118o.jpg'
URL_TEST3 = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSiMNLCFPvZBb2pqxKcNx_1xTrBKDwzd44hkw&usqp=CAU'
URL_TEST4 = 'https://agfight.com/wp-content/uploads/2019/07/gadelha-1.jpg'
URL_TEST5 = 'https://www.superlutas.com.br/wp-content/uploads/2020/02/dominick-reyes-ufc247-1.jpg'
URL_TEST6 = 'https://www.lowking.pl/wp-content/uploads/2019/11/dawodu_4.jpg'
URL_TEST7 = 'https://i.ytimg.com/vi/wdsH1jK10fI/maxresdefault.jpg'
URL_TEST8 = 'https://sportshub.cbsistatic.com/i/r/2019/11/17/8c67de3c-0fae-4797-8287-0e7ea1c91aae/thumbnail/1200x675/990077a2e32127fd11136366f4a03d08/ufc.jpg'
URL_TEST9 = 'https://www.superlutas.com.br/wp-content/uploads/2019/01/conor-foto-reprodu%C3%A7%C3%A3o-instagram-@thenotoriousmma-e1554302752374.jpg'
URL_TEST10 = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSMz7oueTNIi87pS6NaRQU21iY-GdWixvjAAA&usqp=CAU'

facenet_model = load_model(PATH_TO_LOAD_FACENET_MODEL, compile=False)

detector = MTCNN()


def extract_image_from_path(filename: str) -> np.ndarray:

    #file = os.path.join(global_image_path, filename)
    image = Image.open(filename).convert('RGB')
    pixels = np.asarray(image)

    return pixels


def extract_face_from_image_array(image_array: np.ndarray) -> np.ndarray:
    faces = detector.detect_faces(image_array)
    x1, y1, width, height = faces[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = image_array[y1:y2, x1:x2]

    return face


def resize_extracted_face(image_array: np.ndarray, required_size: Tuple[int, int] = (160, 160)) -> np.ndarray:
    image = Image.fromarray(image_array).resize(required_size)
    face_array = asarray(image)

    return face_array


def extract_face(filename: str, required_size: Tuple[int, int] = (160, 160)) -> np.ndarray:
    image = extract_image_from_path(filename)
    face = extract_face_from_image_array(image)
    resized_face = resize_extracted_face(face)

    return resized_face


def load_faces(directory: str) -> List[np.ndarray]:
    """ Load images and extract faces for all images in a directory """

    faces = list()

    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + '/' + filename

        # get face
        face = extract_face(path)

        # store
        faces.append(face)

    return faces


def load_dataset(directory: str) -> (List[np.ndarray], np.ndarray):
    """ Load a dataset that contains one subdir for each class that in turn contains images """

    X, y = list(), list()

    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + '/' + subdir + '/'

        # skip any files that might be in the dir
        if not isdir(path):
            continue

        # load all faces in the subdirectory
        faces = load_faces(path)

        # create labels
        labels = [subdir for _ in range(len(faces))]

        # store
        X.extend(faces)
        y.extend(labels)

    return asarray(X), asarray(y)


def create_train_test_data():

    # load train dataset
    trainX, trainy = load_dataset(TRAIN_SET_PATH)

    # load test dataset
    testX, testy = load_dataset(TEST_SET_PATH)

    # save arrays to one file in compressed format
    return trainX, trainy, testX, testy


def get_embedding(facenet_model, face_pixels):
    """ Get the face embedding for one face """

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


def create_embedding_data_set(trainX, trainy, testX, testy):
    # convert each face in the train set to an embedding
    newTrainX = list()
    for face_pixels in trainX:
        embedding = get_embedding(facenet_model, face_pixels)
        newTrainX.append(embedding)
    newTrainX = asarray(newTrainX)

    # convert each face in the test set to an embedding
    newTestX = list()
    for face_pixels in testX:
        embedding = get_embedding(facenet_model, face_pixels)
        newTestX.append(embedding)
    newTestX = asarray(newTestX)

    return newTrainX, trainy, newTestX, testy


def normalize_input_vector(trainX, testX):
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)

    return trainX, testX


def enconde_targets(trainy, testy):
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)

    return trainy, testy, out_encoder


def train_model(trainX, trainy):

    # fit model
    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)

    return model


def predict_fighters(model, testX):
    # predict
    yhat_test = model.predict(testX)

    return yhat_test


def get_image_array_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    pixels = np.asarray(img)
    return pixels


def extract_face_from_url(url):
    image_array = get_image_array_from_url(url)
    face_array = extract_face_from_image_array(image_array)
    resized_face_array = resize_extracted_face(face_array)

    return resized_face_array


def normalize_one_array(image_array):
    in_encoder = Normalizer(norm='l2')
    normalized_array = in_encoder.transform(image_array)

    return normalized_array


def test_model_on_selected_photo(url, model, out_encoder):
    face_array = extract_face_from_url(url)
    face_embedding = get_embedding(facenet_model, face_array)
    face_embedding = expand_dims(face_embedding, axis=0)
    face_embedding_normalized = normalize_one_array(face_embedding)

    #sample = expand_dims(face_embedding_normalized, axis = 0)

    yhat_class = model.predict(face_embedding_normalized)
    yhat_prob = model.predict_proba(face_embedding_normalized)

    #photo_name = out_encoder.inverse_transform([photo_class])

    class_index = yhat_class[0]
    class_probability = yhat_prob[0, class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)

    # plot for fun
    plt.imshow(face_array)
    plt.title("Predicted: {0} Probability {1:.2f}".format(
        predict_names[0], class_probability))
    plt.show()


def create_embedding_data_set_and_train_model():
    trainX, trainy, testX, testy = create_train_test_data()
    trainX, trainy, testX, testy = create_embedding_data_set(trainX, trainy,
                                                             testX, testy)
    trainX, testX = normalize_input_vector(trainX, testX)
    trainy, testy, out_encoder = enconde_targets(trainy, testy)

    model = train_model(trainX, trainy)

    return model, out_encoder


def create_data_set_for_hyperparameter_tuning():
    trainX, trainy, testX, testy = create_train_test_data()
    trainX, trainy, testX, testy = create_embedding_data_set(trainX, trainy,
                                                             testX, testy)
    trainX, testX = normalize_input_vector(trainX, testX)
    trainy, testy, out_encoder = enconde_targets(trainy, testy)

    return trainX, trainy, testX, testy, out_encoder


def training_evaluation(trainX, trainy, cv=10, score='accuracy'):
    scores = cross_val_score(SVC(kernel='linear'), trainX, trainy,
                             scoring=score, cv=cv)
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


def fine_tune_model(trainX, trainy, cv=5):
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': ['auto', 'scale'],
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                  'probability': [True]}

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1,
                        return_train_score=True, cv=cv)

    # fitting the model for grid search
    grid.fit(trainX, trainy)
    #model = grid.best_estimator_
    #model = grid.best_estimator_.set_params(probability=True)

    return grid.best_estimator_


def model_accuracy(model, testX, testy, out_encoder):
    yhat = predict_fighters(model, testX)
    print(confusion_matrix(testy, yhat))
    print()
    print(out_encoder.classes_)


def svm_confusion_matrix(model, testX, testy, out_encoder):
    yhat = predict_fighters(model, testX)
    print(pd.crosstab(testy, yhat, rownames=['Real'],
                      colnames=['     Predicted'], margins=True))


def model_metrics(model, testX, testy, out_encoder):
    yhat = predict_fighters(model, testX)
    print(metrics.classification_report(testy, yhat))

#trainX, trainy, testX, testy, out_encoder = create_data_set_for_hyperparameter_tuning()

#model = fine_tune_model(trainX, trainy)


def save_pickle(model, out_encoder):
    pickle.dump(model, open(UFC_PROJECT_PATH + TRAINED_MODEL, 'wb'))
    pickle.dump(out_encoder, open(
        UFC_PROJECT_PATH + TRAINED_OUT_ENCODER, 'wb'))

#save_pickle(model, out_encoder)


def load_pickle():
    load_model = pickle.load(open(UFC_PROJECT_PATH + TRAINED_MODEL, 'rb'))
    load_out_encoder = pickle.load(
        open(UFC_PROJECT_PATH + TRAINED_OUT_ENCODER, 'rb'))

    return load_model, load_out_encoder


def saving_compressed_array(trainX, trainy, testX, testy):
    save_compressed_array = savez_compressed(
        SAVE_NPZ + 'ufc-253-fighters-face-dataset.npz', trainX, trainy, testX, testy)

#saving_compressed_array(trainX, trainy, testX, testy)


def two_fighters_accuracy(model, out_encoder):
    data = load(SAVE_NPZ + 'ufc-253-fighters-face-dataset.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print("Training evaluation")
    training_evaluation(trainX, trainy)
    print('\n')
    print('Confusion Matrix')
    svm_confusion_matrix(model, testX, testy, out_encoder)
    model_metrics(model, testX, testy, out_encoder)
    print(out_encoder.classes_)


def load_trained_model():
    try:
        model, out_encoder = load_pickle()
    except Exception:
        print('Training model...')
        trainX, trainy, testX, testy, out_encoder = create_data_set_for_hyperparameter_tuning()
        model = fine_tune_model(trainX, trainy)
        save_pickle(model, out_encoder)
        saving_compressed_array(trainX, trainy, testX, testy)
    return model, out_encoder


def main():

    model, out_encoder = load_trained_model()

    two_fighters_accuracy(model, out_encoder)

    test_model_on_selected_photo(URL_TEST1, model, out_encoder)

    test_model_on_selected_photo(URL_TEST2, model, out_encoder)

    test_model_on_selected_photo(URL_TEST3, model, out_encoder)

    test_model_on_selected_photo(URL_TEST4, model, out_encoder)

    test_model_on_selected_photo(URL_TEST5, model, out_encoder)

    test_model_on_selected_photo(URL_TEST6, model, out_encoder)

    test_model_on_selected_photo(URL_TEST7, model, out_encoder)


if __name__ == "__main__":
    main()
