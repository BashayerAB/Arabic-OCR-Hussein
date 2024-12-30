import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import cv2
import os
from tqdm import tqdm
from myModel import EfficientNetRNNModel

chars = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف',
         'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي', 'لا']

def read_data(dataset_path, image_size=(25, 25)):
    X = []
    Y = []
    for char in tqdm(os.listdir(dataset_path)):
        char_folder = os.path.join(dataset_path, char)
        if os.path.isdir(char_folder):
            for image_path in glob(f"{char_folder}/*.png"):
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                img_resized = cv2.resize(img, image_size)
                X.append(img_resized)
                Y.append(char)
    return np.array(X), np.array(Y)


checkpoint_path = "/content/drive/MyDrive/Saved_Models/new_save_model_rnn_eff_92.pth"
model = EfficientNetRNNModel(num_classes=len(chars))  # Adjust num_classes to your character set size
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Move model to GPU if available

if os.path.exists(checkpoint_path):
    print("Loading pretrained weights...")
    model.load_state_dict(torch.load(checkpoint_path))
else:
    print("Pretrained model not found. Training from scratch.")


def train():
    dataset_path = '/content/Dhad/Dhad_Dataset'
    X, Y = read_data(dataset_path)
    X = torch.tensor(X).float().unsqueeze(1)  # Add channel dimension
    Y = torch.tensor([chars.index(y) for y in Y])  # Convert labels to indices

    # Train/test split
    train_size = int(0.8 * len(X))
    X_train, Y_train = X[:train_size], Y[:train_size]
    X_test, Y_test = X[train_size:], Y[train_size:]

    # DataLoader
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=32, shuffle=False)

    # Model, loss, and optimizer
    model = EfficientNetRNNModel(num_classes=len(chars))
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(10):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to('cuda'), batch_y.to('cuda')

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader)}')

    # Save model
    torch.save(model.state_dict(), '/content/drive/MyDrive/Saved_Models/new_model.pth')

if __name__ == "__main__":
    train()
