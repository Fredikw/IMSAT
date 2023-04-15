"""
Libraries

"""
import torch

import torch.nn as nn
import torch.nn.init as init

"""
Feed Forward Neural Network

"""

class NeuralNet(nn.Module):
    def __init__(self, num_classes=121):
        super(NeuralNet, self).__init__()
        
        # Add first fully connected layer with 428 * 428 input neurons and 1200 output neurons
        self.fc1 = nn.Linear(428 * 428, 1200)
        # Initialize the weights of the first fully connected layer using the He normal initialization
        init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        # Add first batch normalization layer with 1200 neurons and epsilon = 2e-5
        self.bn1   = nn.BatchNorm1d(1200, eps=2e-5)
        self.bn1_F = nn.BatchNorm1d(1200, eps=2e-5, affine=False)
        # Add first ReLU activation function
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(1200, 1200)
        init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        self.bn2   = nn.BatchNorm1d(1200, eps=2e-5)
        self.bn2_F = nn.BatchNorm1d(1200, eps=2e-5, affine=False)

        self.relu2 = nn.ReLU()
        
        # Add output layer of size 10 
        self.fc3 = nn.Linear(1200, num_classes)
        init.kaiming_normal_(self.fc3.weight, nonlinearity='linear')
        
    # Define the forward pass through the network
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        # Pass the input through the first fully connected layer
        x = self.fc1(x)
        # Pass the output of the first fully connected layer through the first batch normalization layer
        x = self.bn1(x)
        # Pass the output of the first batch normalization layer through the first ReLU activation function
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        return x

"""
Convolutional Neural Network

"""

class CNN(nn.Module):
    def __init__(self, num_classes=121):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Define max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Define dropout layer
        self.dropout = nn.Dropout(p=0.5)
        # Define fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 107 * 107, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)
            
    def forward(self, x):
        # Apply convolutional layer 1, activation function (ReLU), and max pooling
        x = self.pool(nn.functional.relu(self.conv1(x)))
        # Apply convolutional layer 2, activation function (ReLU), and max pooling
        x = self.pool(nn.functional.relu(self.conv2(x)))
        # Flatten the output for the fully connected layers
        x = x.view(-1, 64 * 107 * 107)  # -1 is a placeholder for the batch size
        # Apply fully connected layer 1 and activation function (ReLU)
        x = nn.functional.relu(self.fc1(x))
        # Apply dropout layer
        x = self.dropout(x)
        # Apply fully connected layer 2
        x = self.fc2(x)
        
        return x