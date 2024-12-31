##########last train.py
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from glob import glob
from tqdm import tqdm
from myModel import EfficientNetRNNModel  # Import your custom model
checkpoint_path = "/content/drive/MyDrive/Saved_Models/new_save_model_rnn_eff_92.pth"
model = EfficientNetRNNModel(num_classes=29)  # Adjust num_classes to match your character set size
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
if os.path.exists(checkpoint_path):
    print("Loading pretrained weights...")
    state_dict = torch.load(checkpoint_path)
    
    # Filter the state_dict to include only keys that exist in the current model
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(filtered_state_dict, strict=False)  # Allow partial loading
else:
    print("Pretrained model not found. Training from scratch.")
# Load pretrained model
#checkpoint_path = "/content/drive/MyDrive/Saved_Models/new_save_model_rnn_eff_92.pth"
#model = EfficientNetRNNModel(num_classes=29)  # Adjust num_classes to match your character set size
#model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

#if os.path.exists(checkpoint_path):
    #print("Loading pretrained weights...")
    #state_dict = torch.load(checkpoint_path)
    #model.load_state_dict(state_dict, strict=False)
    # Remove `classifier` keys
    #filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}
    #model.load_state_dict(filtered_state_dict, strict=False)
#else:
    #print("Pretrained model not found. Training from scratch.")

# Define optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def read_data(dataset_path, image_size=(25, 25)):
    X = []
    Y = []
    chars = sorted(os.listdir(dataset_path))  # Ensure consistent indexing of character classes
    char_to_index = {char: idx for idx, char in enumerate(chars)}
  
    print(f"Loading data from {dataset_path}...")
    for char in tqdm(chars, total=len(chars)):
        char_folder = os.path.join(dataset_path, char)
        if os.path.isdir(char_folder):
            images = glob(f"{char_folder}/*.png")
            for image_path in images:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                img_resized = cv2.resize(img, image_size)

                # Convert grayscale to 3-channel RGB
                img_rgb = cv2.merge([img_resized, img_resized, img_resized])

                X.append(img_rgb)
                Y.append(char_to_index[char])  # Convert character to index

    X = np.array(X).reshape(-1, 3, *image_size) / 255.0  # Normalize images
    Y = np.array(Y)
    return X, Y


# Define `num_classes` after determining the characters
chars = sorted(os.listdir("/content/Dhad/Dhad_Dataset/train"))  # Character folders
num_classes = len(chars)  # Total number of classes

# Updated model initialization to use `num_classes`
model = EfficientNetRNNModel(num_classes=num_classes)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Update dataset validation
def train():
    # Paths to train and test directories
    train_dataset_path = "/content/Dhad/Dhad_Dataset/train"
    test_dataset_path = "/content/Dhad/Dhad_Dataset/test"

    # Load training and testing data
    X_train, Y_train = read_data(train_dataset_path)
    X_test, Y_test = read_data(test_dataset_path)

    # Validate dataset labels
    print("Unique train labels:", np.unique(Y_train))
    print("Unique test labels:", np.unique(Y_test))
    
    # Ensure labels are in the range [0, num_classes-1]
    if np.any(Y_train < 0) or np.any(Y_train >= num_classes):
        raise ValueError(f"Train labels are out of range [0, {num_classes-1}]")

    if np.any(Y_test < 0) or np.any(Y_test >= num_classes):
        raise ValueError(f"Test labels are out of range [0, {num_classes-1}]")

    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(Y_train).long())
    test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(Y_test).long())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Training loop
    for epoch in range(10):  # Adjust the number of epochs
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            print("Inputs shape:", inputs.shape)
            print("Labels shape:", labels.shape)

            # Ensure inputs have 3 channels and expected image size
            if inputs.shape[1] != 3:
                raise ValueError(f"Expected 3 channels, but got {inputs.shape[1]} channels.")

            # Ensure labels match the batch size
            if labels.shape[0] != inputs.shape[0]:
                raise ValueError(f"Mismatch between input batch size {inputs.shape[0]} and labels batch size {labels.shape[0]}.")

            #break  # Check only the first batch

    # Save trained model
    torch.save(model.state_dict(), "/content/drive/MyDrive/Saved_Models/trained_effnet_rnn.pth")
    print("Model saved successfully!")


if __name__ == "__main__":
    train()
