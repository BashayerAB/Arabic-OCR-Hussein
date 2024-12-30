import torch.nn as nn
import torchvision.models as models

class EfficientNetRNNModel(nn.Module):
    def __init__(self, num_classes, rnn_hidden_size=128, num_rnn_layers=1, nonlinearity='tanh'): #Added the nonlinearity argument to choose between tanh and relu (default is tanh).
        super(EfficientNetRNNModel, self).__init__()

        # Load EfficientNet B0 as feature extractor
        efficientnet = models.efficientnet_b0(pretrained=True)

        # Remove the last fully connected layer, we only need the feature extractor
        self.feature_extractor = nn.Sequential(*list(efficientnet.children())[:-2])

        # RNN layers
        self.rnn = nn.RNN(input_size=1280, hidden_size=rnn_hidden_size, num_layers=num_rnn_layers,
                          batch_first=True, nonlinearity=nonlinearity)

        # Fully connected layer
        self.fc = nn.Linear(rnn_hidden_size, num_classes)

    def forward(self, x):
        # EfficientNet feature extractor
        x = self.feature_extractor(x)

        # Reshape for RNN: (batch_size, channels, height, width) -> (batch_size, channels, height * width)
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, height * width)  # (batch_size, channels, height * width)

        # Transpose the dimensions to match RNN input: (batch_size, channels, height * width) -> (batch_size, height * width, channels)
        x = x.permute(0, 2, 1)  # (batch_size, height * width, channels)

        # RNN forward pass
        x, _ = self.rnn(x)  # x: (batch_size, height * width, hidden_size)

        # Select the last time step output
        x = x[:, -1, :]  # (batch_size, hidden_size)

        # Fully connected layer to get final class scores
        x = self.fc(x)  # (batch_size, num_classes)

        return x

