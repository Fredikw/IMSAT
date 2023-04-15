"""
Libraries

"""

import torch.nn as nn
import torch.nn.init as init


"""
Feed Forward Neural Network

"""

class NeuralNet(nn.Module):
    def __init__(self):
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
        self.fc3 = nn.Linear(1200, 121)
        init.kaiming_normal_(self.fc3.weight, nonlinearity='linear')
        
    # Define the forward pass through the network
    def forward(self, x):
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