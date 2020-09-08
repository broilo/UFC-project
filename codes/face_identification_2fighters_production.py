from sklearn.metrics import roc_curve, roc_auc_score
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
from sklearn.model_selection import cross_val_predict
import seaborn as sns


TRAIN_SET_PATH = r"C:\Users\thiag\OneDrive\Área de Trabalho\virtual_environments\ufc_project\ufc_2fighters\2-ufc-fighters\train"
TESTE_SET_PATH = r'C:\Users\thiag\OneDrive\Área de Trabalho\virtual_environments\ufc_project\ufc_2fighters\2-ufc-fighters\test'
PATH_TO_LOAD_FACENET_MODEL = r"C:\Users\thiag\OneDrive\Área de Trabalho\virtual_environments\ufc_project\ufc_2fighters\FaceNet\model\facenet_keras.h5"
FACES_PATH = '2-ufc-fighters/'
TRAINED_MODEL = 'trained_model.sav'
TRAINED_OUT_ENCODER = 'trained_out_encoder.sav'
URL_TEST = 'https://i.pinimg.com/originals/8b/95/b5/8b95b5db2d2b315dd75ddeddfe388538.jpg'
UFC_PROJECT_PATH = r'C:\Users\thiag\OneDrive\Área de Trabalho\virtual_environments\ufc_project\ufc_2fighters'

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


def load_dataset(directory: str) \
        -> (List[np.ndarray], np.ndarray):
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
    testX, testy = load_dataset(TESTE_SET_PATH)

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
    plt.title("Predicted: {0} Probability {1:.3f}".format(
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


def crate_data_set_for_hyperparameter_tuning():
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

    grid.fit(trainX, trainy)

    return grid.best_estimator_


def model_accuracy(model, testX, testy, out_encoder):
    yhat = predict_fighters(model, testX)

    labels = out_encoder.classes_.tolist()
    cm = confusion_matrix(testy, yhat)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax)

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Conor', 'Khabib'])
    ax.yaxis.set_ticklabels(['Conor', 'Khabib'])
    plt.show()


def load_pickle():
    load_model = pickle.load(open(UFC_PROJECT_PATH + TRAINED_MODEL, 'rb'))
    load_out_encoder = pickle.load(
        open(UFC_PROJECT_PATH + TRAINED_OUT_ENCODER, 'rb'))

    return load_model, load_out_encoder


def save_pickle(model, out_encoder):
    pickle.dump(model, open(UFC_PROJECT_PATH + TRAINED_MODEL, 'wb'))
    pickle.dump(out_encoder, open(
        UFC_PROJECT_PATH + TRAINED_OUT_ENCODER, 'wb'))


def plot_roc_curve(fpr, tpr, trainy, y_scores, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    auc_score = "ROC AUC = {:.4f}".format(
        roc_auc_score(trainy, y_scores[:, 1]))
    plt.annotate(auc_score, (0.5, 0.3))
    plt.show()


def classifier_evaluation(trainX, trainy):
    y_scores = cross_val_predict(fine_tune_model(
        trainX, trainy), trainX, trainy, cv=3, method="predict_proba")
    y_scores_svm = y_scores[:, 1]  # score = proba of positive class
    fpr, tpr, thresholds_svm = roc_curve(trainy, y_scores_svm)
    plot_roc_curve(fpr, tpr, trainy, y_scores)


def two_fighters_accuracy(model, out_encoder):
    data = load(
        r'C:\Users\thiag\OneDrive\Área de Trabalho\virtual_environments\ufc_project\ufc_2fighters\2-fighters_face_dataset.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print("Training evaluation")
    training_evaluation(trainX, trainy)
    print('\n')
    model_accuracy(model, testX, testy, out_encoder)
    classifier_evaluation(trainX, trainy)


def saving_compressed_array(trainX, trainy, testX, testy):
    savez_compressed(
        r'C:\Users\thiag\OneDrive\Área de Trabalho\virtual_environments\ufc_project\ufc_2fighters\2-fighters_face_dataset.npz', trainX, trainy, testX, testy)


def load_trained_model():
    try:
        model, out_encoder = load_pickle()
    except Exception:
        print('Training model...')
        trainX, trainy, testX, testy, out_encoder = crate_data_set_for_hyperparameter_tuning()
        model = fine_tune_model(trainX, trainy)
        save_pickle(model, out_encoder)
        saving_compressed_array(trainX, trainy, testX, testy)
    return model, out_encoder


def main():

    model, out_encoder = load_trained_model()

    two_fighters_accuracy(model, out_encoder)

    test_model_on_selected_photo(
        'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSBbmAcIU9L8RqR0c4UhIzvPNxb04vNVj0MQw&usqp=CAU', model, out_encoder)

    test_model_on_selected_photo(
        'https://imagez.tmz.com/image/33/1by1/2016/11/15/33b9ebee31bc5646938781305ff0cfe7_xl.jpg', model, out_encoder)


if __name__ == '__main__':
    main()
