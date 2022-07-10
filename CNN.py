import torch
from torch import nn
from torch._C import device
import hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork_Recurrent(nn.Module):
    def __init__(self, actionSpaceSize, obsSpaceSize):
        self.actionSpaceSize = actionSpaceSize
        self.obsSpaceSize = obsSpaceSize
        super(NeuralNetwork_Recurrent, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 8, 4)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        torch.nn.init.kaiming_uniform_(self.conv3.weight)

        self.fc1 = nn.Linear(2560, 512)
        self.lstm = nn.LSTM(input_size = 2560, hidden_size = 512, batch_first = False)
        #torch.nn.init.kaiming_uniform_(self.fc1.weight)
        # Output 2 values: fly up and do nothing
        self.fc2 = nn.Linear(512, self.actionSpaceSize)
        self.fc2_ = nn.Linear(512, 1)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc2_.weight)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, hidden, cell):
        x = x.to(device)
        hidden = hidden.to(device)
        cell = cell.to(device)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        # Flatten output to feed into fully connected layers
        x = x.view(x.size()[0], -1)
        x = torch.unsqueeze(x,  axis = 0)
        #x = self.relu(self.fc1(x))
        x , (next_hidden, next_cell) = self.lstm(x, (hidden, cell))
        x = self.fc2(x)
        return x, next_hidden, next_cell

    def forward2(self, x):
        x = torch.squeeze(x, axis = 0)
        x = x.to(device)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        # Flatten output to feed into fully connected layers
        x = x.view(x.size()[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2_(x)
        return x

class NeuralNetwork_Forward(nn.Module):
    def __init__(self, actionSpaceSize, obsSpaceSize):
        self.actionSpaceSize = actionSpaceSize
        self.obsSpaceSize = obsSpaceSize
        super(NeuralNetwork_Forward, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        torch.nn.init.kaiming_uniform_(self.conv3.weight)

        self.fc1 = nn.Linear(2560, 512)
        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        # Output 2 values: fly up and do nothing
        self.fc2 = nn.Linear(512, self.actionSpaceSize)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.squeeze(x, axis = 0)
        x = x.to(device)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        # Flatten output to feed into fully connected layers
        x = x.view(x.size()[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x