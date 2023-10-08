import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, device, input_size, hidden_size, num_layers, sequence_length, num_classes, *args, **kwargs) -> None:
        super(RNN, self).__init__(*args, **kwargs)
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) # num_layers, batch_size, hiddden size
        out, _ = self.rnn(x, h0) # out, hidden_state
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

class GRU_RNN(nn.Module):
    def __init__(self, device, input_size, hidden_size, num_layers, sequence_length, num_classes, *args, **kwargs) -> None:
        super(GRU_RNN, self).__init__(*args, **kwargs)
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) # num_layers, batch_size, hiddden size
        out, _ = self.gru(x, h0) # out, hidden_state
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

class LSTM(nn.Module):
    def __init__(self, device, input_size, hidden_size, num_layers, sequence_length, num_classes, *args, **kwargs) -> None:
        super(LSTM, self).__init__(*args, **kwargs)
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) # num_layers, batch_size, hiddden size
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.gru(x, (h0, c0)) # out, hidden_state
        out = self.fc(out[:,-1,:])
        return out