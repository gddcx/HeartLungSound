import torch.nn as nn
import torch.nn.functional as F

class Layer1(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru1 = nn.GRU(input_size=40, hidden_size=32, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(input_size=32, hidden_size=128, num_layers=1, batch_first=True)

        self.gru3 = nn.GRU(input_size=40, hidden_size=64, num_layers=1, batch_first=True)
        self.gru4 = nn.GRU(input_size=64, hidden_size=128, num_layers=1, batch_first=True)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, data):
        print(data.shape)
        out = self.gru1(data)[0]
        out1 = F.leaky_relu(out, negative_slope=1e-2)
        # out1 = out
        out1 = self.gru2(out1)[0]
        out1 = self.leaky_relu(out1)

        out2 = self.gru3(data)[0]
        out2 = self.leaky_relu(out2)
        out2 = self.gru4(out2)[0]
        out2 = self.leaky_relu(out2)
        return out, out1+out2

class Layer2(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru5 = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        self.gru6 = nn.GRU(input_size=128, hidden_size=32, num_layers=1, batch_first=True)

        self.gru7 = nn.GRU(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        self.gru8 = nn.GRU(input_size=64, hidden_size=32, num_layers=1, batch_first=True)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, data):
        out1 = self.gru5(data)[0]
        out1 = self.leaky_relu(out1)
        out1 = self.gru6(out1)[0]
        out1 = self.leaky_relu(out1)

        out2 = self.gru7(data)[0]
        out2 = self.leaky_relu(out2)
        out2 = self.gru8(out2)[0]
        out2= self.leaky_relu(out2)
        return out1+out2

class Layer3(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(in_features=32*39, out_features=64*39)
        self.dense2 = nn.Linear(in_features=64*39, out_features=16*10)

        self.dense3 = nn.Linear(in_features=32*39, out_features=32*39)
        self.dense4 = nn.Linear(in_features=32*39, out_features=16*10)

        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        out1 = self.dense1(data)
        out1 = self.leaky_relu(out1)
        out1 = self.dropout(out1)
        out1 = self.dense2(out1)
        out1 = self.leaky_relu(out1)

        out2 = self.dense3(data)
        out2 = self.leaky_relu(out2)
        out2 = self.dropout(out2)
        out2 = self.dense4(out2)
        out2 = self.leaky_relu(out2)
        return out1+out2

class ModelApproch2(nn.Module):
    def __init__(self):
        super().__init__()
        # self.layer1 = Layer1()
        # self.layer2 = Layer2()
        # self.layer3 = Layer3()
        self.gru1 = nn.GRU(hidden_size=10, input_size=40, batch_first=True, num_layers=2)

        self.dense1 = nn.Linear(in_features=39*10, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=6)

    def forward(self, data):
        out = self.gru1(data)[0]
        out = out.flatten(start_dim=1)
        out = self.dense1(out)
        out = F.leaky_relu(out, negative_slope=1e-2)
        out = F.dropout(out, p=0.5)
        out = self.dense2(out)
        return out