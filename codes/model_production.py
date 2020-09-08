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
import pandas as pd


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
    """Receives a path to an image, open it and convert it to RGB. Finally, 
    returns it as a numpy array.

    Args:
        filename (str): path to the photo in disk.

    Returns:
        np.ndarray: photo in a numpy array format.
    """

    image = Image.open(filename).convert('RGB')
    pixels = np.asarray(image)

    return pixels


def extract_face_from_image_array(image_array: np.ndarray) -> np.ndarray:
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
    x1, y1, width, height = faces[0]['box']
    # Fixing a bug, since sometimes the positions may come as negative numbers
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = image_array[y1:y2, x1:x2]

    return face


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


def extract_face(filename: str, required_size: Tuple[int, int] = (160, 160)) -> np.ndarray:
    """Receives a file path to an image and then call extract_image_from_path to
    get the image array. After, calls extract_face_from_image_array to make use 
    of the neural network MCTNN and store in face just the face contained in the
     photo. And then, finally, resize the image.

    Args:
        filename (str): path to the photo in disk.
        required_size (Tuple[int, int], optional): Image size. Defaults to 
        160, 160).

    Returns:
        np.ndarray: Resized image array containing only the face.
    """
    image = extract_image_from_path(filename)
    face = extract_face_from_image_array(image)
    resized_face = resize_extracted_face(face)

    return resized_face


def load_faces(directory: str) -> List[np.ndarray]:
    """Receives a path to a directory, and the extract all faces in it. It 
    returns a list containg all faces in the directory.

    Args:
        directory (str): Path to the directory containing the faces.

    Returns:
        List[np.ndarray]: list containing array with all the faces in the 
        directory.
    """

    faces = list()

    # enumerate files
    for filename in listdir(directory):
        path = directory + '/' + filename
        face = extract_face(path)
        faces.append(face)

    return faces


def load_dataset(directory: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load a dataset that contains one subdir for each class that in turn 
    contains images. After extract faces for each one, creates an array with 
    labels.

    Args:
        directory (str): Path to the directory containing the faces.

    Returns:
        List[np.ndarray]: list containing array with all the faces in the 
        directory.
        np.ndarray: array containg all the labels.
    """

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


def create_train_test_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create all the data set to be used in training, using the method 
    load_dataset. It returns train and test arrays.

    Returns:
        np.ndarray, np.ndarray, np.ndarray, np.ndarray: train array, train array 
        labels, test array and test array labels.
    """

    # load train dataset
    trainX, trainy = load_dataset(TRAIN_SET_PATH)

    # load test dataset
    testX, testy = load_dataset(TESTE_SET_PATH)

    # save arrays to one file in compressed format
    return trainX, trainy, testX, testy


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


def create_embedding_data_set(trainX: np.ndarray, trainy: np.ndarray, testX: np.ndarray, testy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Given the array representation of faces and labels dataset, returns the 
    emdedding representation.

    Args:
        trainX (np.ndarray): train array.
        trainy (np.ndarray): train array labels.
        testX (np.ndarray): test array.
        testy (np.ndarray): test array labels.

    Returns:
        np.ndarray, np.ndarray, np.ndarray, np.ndarray: embedding train array, 
        train array labels, embedding test array and test array labels.
    """
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


def normalize_input_vector(trainX: np.ndarray, testX: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize the input vector

    Args:
        trainX (np.ndarray): train embedding array.
        testX (np.ndarray): test embedding array.

    Returns:
        np.ndarray, np.ndarray: normalized train and test arrays.
    """
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)

    return trainX, testX


def enconde_targets(trainy: np.ndarray, testy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Receives train and test labels, and encode it. Returns encoded arrays and 
    an out_encoder variable that maps fighters names and the encoded numbers.

    Args:
        trainy (np.ndarray): train labels.
        testy (np.ndarray): test labels.

    Returns:
        np.ndarray, np.ndarray, LabelEncoder: encoded labels and variable that 
        store the mapping between names and encoders.
    """
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)

    return trainy, testy, out_encoder


def predict_fighters(model: SVC, testX: np.ndarray) -> np.ndarray:
    """Given the model and the test set, predict the fighter name.

    Args:
        model (SVC): Machine Learning model used do classify the fighters.
        testX (np.ndarray): test array containg embedding images.

    Returns:
        np.ndarray: predictions done by the model.
    """
    # predict
    yhat_test = model.predict(testX)

    return yhat_test


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

def get_information(name: str):
    """Receives a fighter name, and then make a request to get information about
    it in ufc repository.

    Args:
        name (str): fighters name.
    """
    df = pd.read_csv("C:\\Users\\thiag\\OneDrive\\Área de Trabalho\\virtual_environments\\ufc_project\\ufc_2fighters\\fighters_info.csv")
    name = name.split('-')
    name = name[0] + ' ' + name[1]
    print(df[df['Nome'] == name])


# def get_fights_history(name: str):


def test_model_on_selected_photo(url: str, model: SVC, out_encoder: LabelEncoder):
    """Receives a url photo, process it and make a prediction using SVC model.

    Args:
        url (str): url containing the image.
        model (SVC): Machine Learning model used do classify the fighters.
        out_encoder (LabelEncoder): variable that store the mapping between 
        names and encoders.
    """
    # Process the url to get and embedding representation of the face.
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
        else:
            get_information(predict_names[0])
            #get_fight_history()
        
        # Plot the result
        plt.imshow(face_array)
        plt.title("Predicted: {0} Probability {1:.3f}".format(
            predict_names[0], class_probability))
        plt.show()
    # Not sure what will be the best return values for the deployment.
    #return predict_names[0], class_probability


def crate_data_set_for_hyperparameter_tuning() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """Process the dataset to be trained.

    Returns:
        np.ndarray, np.ndarray, np.ndarray, np.ndarray: embedding train array, 
        train array labels, embedding test array, test array labels and Label 
        encoder.
    """
    trainX, trainy, testX, testy = create_train_test_data()
    trainX, trainy, testX, testy = create_embedding_data_set(trainX, trainy,
                                                             testX, testy)
    trainX, testX = normalize_input_vector(trainX, testX)
    trainy, testy, out_encoder = enconde_targets(trainy, testy)

    return trainX, trainy, testX, testy, out_encoder


def training_evaluation(trainX: np.ndarray, trainy: np.ndarray, cv: int = 10, score: str = 'accuracy'):
    """Does cross validation and print scores, mean and standard deviation.

    Args:
        trainX (np.ndarray): train array containg embedding images.
        trainy (np.ndarray): train array containg labels.
        cv (int, optional): Number of folds to apply in cross validation. 
                            Defaults to 10.
        score (str, optional): Score to be calculated. Defaults to 'accuracy'.
    """
    scores = cross_val_score(SVC(kernel='linear'), trainX, trainy,
                             scoring=score, cv=cv)
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


def fine_tune_model(trainX: np.ndarray, trainy: np.ndarray, cv: int = 5) -> SVC:
    """Receives training set and run a grid search to find the best 
    hyperparameters. It returns the best model, already trained.

    Args:
        trainX (np.ndarray): train array containg embedding images.
        trainy (np.ndarray): train array containg labels.
        cv (int, optional): Number of folds to apply in cross validation. 
                            Defaults to 5.

    Returns:
        SVC: Trained model.
    """
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': ['auto', 'scale'],
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                  'probability': [True]}

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1,
                        return_train_score=True, cv=cv)

    grid.fit(trainX, trainy)

    return grid.best_estimator_


def load_pickle() -> Tuple[SVC, LabelEncoder]:
    """Load saved model.

    Returns:
        SVC, LabelEncoder: Trained model and label encoder.
    """
    load_model = pickle.load(open(UFC_PROJECT_PATH + TRAINED_MODEL, 'rb'))
    load_out_encoder = pickle.load(
        open(UFC_PROJECT_PATH + TRAINED_OUT_ENCODER, 'rb'))

    return load_model, load_out_encoder


def save_pickle(model: SVC, out_encoder: LabelEncoder):
    """Save trained model.

    Args:
        model (SVC): trained model.
        out_encoder (LabelEncoder): Label encoder.
    """
    pickle.dump(model, open(UFC_PROJECT_PATH + TRAINED_MODEL, 'wb'))
    pickle.dump(out_encoder, open(
        UFC_PROJECT_PATH + TRAINED_OUT_ENCODER, 'wb'))


def plot_confusion_matrix(model: SVC, testX: np.ndarray, testy: np.ndarray, out_encoder: LabelEncoder):
    """Given a model and a test set, it generates a confusion matrix for 
    evaluation.

    Args:
        model (SVC): trained model.
        testX (np.ndarray):  embedding test array.
        testy (np.ndarray): test array labels.
        out_encoder (LabelEncoder): Label encoder.
    """
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


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, trainy: np.ndarray, y_scores: np.ndarray, label: str = None):
    """Given the false positive rate and true positive rate for different values 
    of threshold, plot a roc curve to evaluate the model. Also calculate the 
    auc_score.

    Args:
        fpr (np.ndarray): false positive rate.
        tpr (np.ndarray): true positive rate.
        trainy (np.ndarray): train array with labels.
        y_scores (np.ndarray): Score used do auc_score calculation.
        label (str, optional): Figura label. Defaults to None.
    """
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    auc_score = "ROC AUC = {:.4f}".format(
        roc_auc_score(trainy, y_scores[:, 1]))
    plt.annotate(auc_score, (0.5, 0.3))
    plt.show()


def classifier_evaluation(trainX: np.ndarray, trainy: np.ndarray):
    """Given training data set, calculate false positive rate, true positive 
    rate, and then plot a roc curve.

    Args:
        trainX (np.ndarray): training embedding data set
        trainy (np.ndarray): training dataset labels.
    """
    y_scores = cross_val_predict(fine_tune_model(
        trainX, trainy), trainX, trainy, cv=3, method="predict_proba")
    y_scores_svm = y_scores[:, 1]  # score = proba of positive class
    fpr, tpr, thresholds_svm = roc_curve(trainy, y_scores_svm)
    plot_roc_curve(fpr, tpr, trainy, y_scores)


def two_fighters_accuracy(model: SVC, out_encoder: LabelEncoder):
    """Given a model and a label encoder, show model evaluaton

    Args:
        model (SVC): trained model.
        out_encoder (LabelEncoder): label encoder.
    """
    data = load(
        r'C:\Users\thiag\OneDrive\Área de Trabalho\virtual_environments\ufc_project\ufc_2fighters\2-fighters_face_dataset.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print("Training evaluation")
    training_evaluation(trainX, trainy)
    print('\n')
    plot_confusion_matrix(model, testX, testy, out_encoder)
    classifier_evaluation(trainX, trainy)


def saving_compressed_array(trainX: np.ndarray, trainy: np.ndarray, testX: np.ndarray, testy: np.ndarray):
    """Save the processed dataset.

    Args:
        trainX (np.ndarray): training embedding data set
        trainy (np.ndarray): training dataset labels.
        trainX (np.ndarray): test embedding data set
        trainy (np.ndarray): test dataset labels.
    """
    savez_compressed(
        r'C:\Users\thiag\OneDrive\Área de Trabalho\virtual_environments\ufc_project\ufc_2fighters\2-fighters_face_dataset.npz', trainX, trainy, testX, testy)


def load_trained_model() -> Tuple[SVC, LabelEncoder]:
    """Load trained model. If it is not trained, train the model. Returns 
    trained model and label encoder.

    Returns:
        SVC, LabelEncoder: Trained model and label encoder.
    """
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

    #two_fighters_accuracy(model, out_encoder)

    #test_model_on_selected_photo(
    #    'https://www.petlove.com.br/images/breeds/192471/profile/original/yorkshire-p.jpg?1532539683', model, out_encoder)

    test_model_on_selected_photo(
        'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTPl3VwrKImZgX0oijn6-ZPtBu9x6Bynh53CQ&usqp=CAU', model, out_encoder)


if __name__ == '__main__':
    main()
