import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from myModel import EfficientNetRNNModel  # Import your model class
from glob import glob

# Load test dataset
def read_data(dataset_path, image_size=(25, 25)):
    X = []
    Y = []
    chars = sorted(os.listdir(dataset_path))  # Ensure consistent indexing of character classes
    char_to_index = {char: idx for idx, char in enumerate(chars)}

    print(f"Loading data from {dataset_path}...")
    for char in chars:
        char_folder = os.path.join(dataset_path, char)
        if os.path.isdir(char_folder):
            images = glob(f"{char_folder}/*.png")
            for image_path in images:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                img_resized = cv2.resize(img, image_size)
                img_rgb = cv2.merge([img_resized, img_resized, img_resized])  # Convert to RGB
                X.append(img_rgb)
                Y.append(char_to_index[char])  # Map characters to indices

    X = np.array(X).reshape(-1, 3, *image_size) / 255.0  # Normalize
    Y = np.array(Y)
    return X, Y

def evaluate_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the class with the highest score
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f"Accuracy on test dataset: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    # Define paths
    test_dataset_path = "/content/Dhad/Dhad_Dataset/test"
    checkpoint_path = "/content/drive/MyDrive/Saved_Models/trained_effnet_rnn.pth"

    # Load the saved state dict
    state_dict = torch.load(checkpoint_path)

    # Remove the extra class weights
    state_dict['classifier.1.weight'] = state_dict['classifier.1.weight'][:29]
    state_dict['classifier.1.bias'] = state_dict['classifier.1.bias'][:29]

    # Load test data
    X_test, Y_test = read_data(test_dataset_path)

    # Create DataLoader for the test dataset
    test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(Y_test).long())
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the model
    model = EfficientNetRNNModel(num_classes=29)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the filtered state dict into the model
    print("Loading trained model weights...")
    model.load_state_dict(state_dict, strict=True)

    # Evaluate the model
    evaluate_model(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu')