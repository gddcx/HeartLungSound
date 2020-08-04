import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.VGG_model = torch.hub.load('harritaylor/torchvggish', 'vggish', preprocess=False)
        self.VGG_model.eval()
        self.bigru = nn.GRU(input_size=128 ,hidden_size=128, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(in_features=256, out_features=6)

    def forward(self, data):
        print(data.shape)
        x = self.VGG_model(data)
        print(x.shape)
        x = x.detach()
        x = x.reshape(-1, 10, x.shape[-1])
        print(x.shape)
        x, _ = self.bigru(x)
        # 取最后一个时间步
        x = x[:, -1, :]
        x = self.linear(x)
        return x

