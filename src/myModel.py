############################## my Model.py
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from torchvision import transforms

chars = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي']

class EfficientNetLSTMModel(nn.Module):
    def __init__(self, num_classes, lstm_hidden_size=128, num_lstm_layers=1):
        super(EfficientNetLSTMModel, self).__init__()

        # Load EfficientNet B0 as feature extractor
        efficientnet = models.efficientnet_b0(pretrained=True)

        # Remove the last fully connected layer, we only need the feature extractor
        self.feature_extractor = nn.Sequential(*list(efficientnet.children())[:-2])

        # LSTM layers
        self.lstm = nn.LSTM(input_size=1280, hidden_size=lstm_hidden_size, num_layers=num_lstm_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
    # EfficientNet feature extractor
      x = self.feature_extractor(x)

    # Check the shape after feature extraction
      #print(f"Shape after EfficientNet feature extractor: {x.shape}")  # Should be [batch_size, 1280, 7, 7]

    # Reshape for LSTM: (batch_size, channels, height, width) -> (batch_size, channels, height * width)
      batch_size, channels, height, width = x.size()
      x = x.view(batch_size, channels, height * width)  # (batch_size, channels, height * width)

    # Transpose the dimensions to match LSTM input: (batch_size, channels, height * width) -> (batch_size, height * width, channels)
      x = x.permute(0, 2, 1)  # (batch_size, height * width, channels)

    # Print shape before LSTM, should now be (batch_size, 49, 1280)
      #print(f"Shape before LSTM: {x.shape}")

    # LSTM forward pass
      x, _ = self.lstm(x)  # x: (batch_size, height * width, hidden_size)

    # Select the last time step output
      x = x[:, -1, :]  # (batch_size, hidden_size)

    # Fully connected layer to get final class scores
      x = self.fc(x)  # (batch_size, num_classes)

      return x



class EfficientNetLSTMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, num_classes=29, lstm_hidden_size=128, num_lstm_layers=1, learning_rate=0.001, epochs=10, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.num_classes = num_classes
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        # Initialize the EfficientNetLSTM model
        self.model = EfficientNetLSTMModel(num_classes=num_classes, lstm_hidden_size=lstm_hidden_size, num_lstm_layers=num_lstm_layers).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])  # Assuming input size 224x224 for EfficientNet

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        # Convert data to PyTorch tensors
        X_tensor = torch.stack([self.transform(img.reshape(25, 25)) for img in X]).to(self.device)
        y_tensor = torch.tensor([chars.index(label) for label in y], dtype=torch.long).to(self.device)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            permutation = torch.randperm(X_tensor.size(0))
            epoch_loss = 0

            for i in range(0, X_tensor.size(0), self.batch_size):
                indices = permutation[i:i+self.batch_size]
                batch_X, batch_y = X_tensor[indices], y_tensor[indices]

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss / len(X_tensor)}")

    def predict(self, X):
        X_tensor = torch.stack([self.transform(img.reshape(25, 25)) for img in X]).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return [chars[pred.item()] for pred in predicted]
