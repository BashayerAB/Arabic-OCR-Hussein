############################# train.py
import torch
import numpy as np
import cv2 as cv
import os
import re
import random
from utilities import projection
from glob import glob
from tqdm import tqdm
from myModel import EfficientNetLSTMClassifier

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.utils import shuffle
from sklearn.model_selection  import train_test_split
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle

####################################################
class ArabicCharDataset(Dataset):
    def __init__(self, char_paths, labels, transform=None):
        self.char_paths = char_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.char_paths)

    def __getitem__(self, idx):
        char_img_path = self.data[idx]
        print(f"Loading image: {char_img_path}")  # Debug the file path
        char_img = cv.imread(char_img_path, cv.IMREAD_GRAYSCALE)
        if char_img is None:
          raise ValueError(f"Failed to load image at path: {char_img_path}")
        ready_char = prepare_char(char_img)  # Resize to (224, 224)
        return ready_char, self.labels[idx]

######################################################
chars = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف',
'ق','ك', 'ل', 'م', 'ن', 'ه', 'و','ي','لا']
train_ratio = 0.8
script_path = os.getcwd()
classifiers = [
    svm.LinearSVC(),
    MLPClassifier(alpha=1e-4, hidden_layer_sizes=(100,), max_iter=1000),
    MLPClassifier(alpha=1e-5, hidden_layer_sizes=(200, 100,), max_iter=1000),
    GaussianNB(),
    EfficientNetLSTMClassifier(epochs=5, batch_size=32, learning_rate=0.001)
]

names = ['LinearSVM', '1L_NN', '2L_NN', 'Gaussian_Naive_Bayes', 'EfficientNetLSTM']
skip = [1, 0, 1, 1, 0]


width = 25
height = 25
dim = (width, height)

def bound_box(img_char):
    HP = projection(img_char, 'horizontal')
    VP = projection(img_char, 'vertical')

    top = -1
    down = -1
    left = -1
    right = -1

    i = 0
    while i < len(HP):
        if HP[i] != 0:
            top = i
            break
        i += 1

    i = len(HP)-1
    while i >= 0:
        if HP[i] != 0:
            down = i
            break
        i -= 1

    i = 0
    while i < len(VP):
        if VP[i] != 0:
            left = i
            break
        i += 1

    i = len(VP)-1
    while i >= 0:
        if VP[i] != 0:
            right = i
            break
        i -= 1

    return img_char[top:down+1, left:right+1]


def binarize(char_img):
    _, binary_img = cv.threshold(char_img, 127, 255, cv.THRESH_BINARY)
    # _, binary_img = cv.threshold(word_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    binary_char = binary_img // 255

    return binary_char


def prepare_char(char_img):
    binary_char = binarize(char_img)
    try:
        char_box = bound_box(binary_char)
        resized = cv.resize(char_box, (224, 224), interpolation=cv.INTER_AREA)
    except:
        resized = np.zeros((224, 224))  # Return a blank image if resizing fails
    return resized


def featurizer(char_img):

    flat_char = char_img.flatten()

    return flat_char


def read_data(limit=4000):

    X = []
    Y = []
    print("For each char")
    for char in tqdm(chars, total=len(chars)):

        folder = f'/content/drive/MyDrive/char_sample/{char}'
        char_paths =  glob(f'/content/drive/MyDrive/char_sample/{char}/*.png')

        if os.path.exists(folder):
            os.chdir(folder)

            print(f'\nReading images for char {char}')
            for char_path in tqdm(char_paths[:limit], total=len(char_paths)):
                num = re.findall(r'\d+', char_path)[0]
                char_img = cv.imread(f'{num}.png', 0)
                ready_char = prepare_char(char_img)
                feature_vector = featurizer(ready_char)
                # X.append(char)
                X.append(feature_vector)
                Y.append(char)

            os.chdir(script_path)

    return X, Y


def train():
    data_dir = '/content/drive/MyDrive/char_sample/'  # Adjust as needed
    batch_size = 32

    # Create DataLoaders
    train_loader, test_loader = create_dataloaders(data_dir, chars, batch_size=batch_size)

    scores = []
    for idx, clf in tqdm(enumerate(classifiers), desc='Classifiers'):
        if not skip[idx]:
            print(f"Training {names[idx]}...")

            if names[idx] == 'EfficientNetLSTM':  # Train only EfficientNetLSTM
                clf.fit(train_loader, test_loader)  # Use DataLoaders for training
            else:
                X_train = []
                Y_train = []
                for X_batch, Y_batch in train_loader:
                    X_train.extend(X_batch.numpy())
                    Y_train.extend(Y_batch)
                X_train = np.array(X_train)
                Y_train = np.array(Y_train)

                clf.fit(X_train, Y_train)

            # Evaluate on test set
            X_test = []
            Y_test = []
            for X_batch, Y_batch in test_loader:
                X_test.extend(X_batch.numpy())
                Y_test.extend(Y_batch)
            X_test = np.array(X_test)
            Y_test = np.array(Y_test)

            score = clf.score(X_test, Y_test)
            scores.append(score)
            print(f"Score of {names[idx]}: {score}")

            # Save the model
            destination = f'models'
            if not os.path.exists(destination):
                os.makedirs(destination)

            location = f'/content/Arabic-OCR-Hussein/src/models/{names[idx]}.sav'
            pickle.dump(clf, open(location, 'wb'))

    # Save report
    with open('/content/Arabic-OCR-Hussein/src/models/report.txt', 'w') as fo:
        for score, name in zip(scores, names):
            fo.writelines(f'Score of {name}: {score}\n')



def test(limit=3000):

    location = f'/content/Arabic-OCR-Hussein/src/models/{names[0]}.sav'
    clf = pickle.load(open(location, 'rb'))

    X = []
    Y = []
    tot = 0
    for char in tqdm(chars, total=len(chars)):

        folder = f'/content/drive/MyDrive/char_sample/{char}'
        char_paths =  glob(f'/content/drive/MyDrive/char_sample/{char}/*.png')


        if os.path.exists(folder):
            os.chdir(folder)

            print(f'\nReading images for char {char}')
            tot += len(char_paths) - limit
            for char_path in tqdm(char_paths[limit:], total=len(char_paths)):
                num = re.findall(r'\d+', char_path)[0]
                char_img = cv.imread(f'{num}.png', 0)
                ready_char = prepare_char(char_img)
                feature_vector = featurizer(ready_char)
                # X.append(char)
                X.append(feature_vector)
                Y.append(char)

            os.chdir(script_path)

    cnt = 0
    for x, y in zip(X, Y):

        c = clf.predict([x])[0]
        if c == y:
            cnt += 1
#####################################################################

def create_dataloaders(data_dir, chars, batch_size=32, limit=4000, transform=None):
    dataset = ArabicCharDataset(data_dir, chars, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

###################################################################

if __name__ == "__main__":

    train()
    # test()