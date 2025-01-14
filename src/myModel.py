import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.optim as optim
from torchvision import transforms
# BaseEstimator and ClassifierMixin for scikit-learn compatibility
from sklearn.base import BaseEstimator, ClassifierMixin

# EfficientNetLSTMModel - this should already be defined in the same file
# Ensure the EfficientNetLSTMModel class is defined before you use it in your code

chars = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف',
'ق','ك', 'ل', 'م', 'ن', 'ه', 'و','ي','لا']
class EfficientNetLSTMModel(nn.Module):
    def __init__(self, num_classes, lstm_hidden_size=128, num_lstm_layers=1):
        super(EfficientNetLSTMModel, self).__init__()

        # Load EfficientNet B0 as feature extractor
        efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

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
    def __init__(self, num_classes=29, lstm_hidden_size=128, num_lstm_layers=1, 
                 learning_rate=0.001, epochs=10, batch_size=32, 
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
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
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])

    def fit(self, train_loader, test_loader=None):
        """
        Train the model using DataLoader objects.
        Args:
            train_loader (DataLoader): DataLoader for training data.
            test_loader (DataLoader, optional): DataLoader for validation data.
        """
        self.model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = torch.tensor([chars.index(label) for label in labels], dtype=torch.long).to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss / len(train_loader)}")

            # Optional validation
            if test_loader:
                self.evaluate(test_loader)

    def evaluate(self, test_loader):
        """
        Evaluate the model on a validation/test DataLoader.
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = torch.tensor([chars.index(label) for label in labels], dtype=torch.long).to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")
        return accuracy

    def predict(self, test_loader):
        """
        Predict the labels for a test DataLoader.
        Args:
            test_loader (DataLoader): DataLoader for test data.
        Returns:
            list: Predicted labels for the entire dataset.
        """
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for images, _ in test_loader:  # Ignore true labels during prediction
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                predictions.extend([chars[pred.item()] for pred in predicted])
        return predictions
