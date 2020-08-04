import torch.nn as nn
import torch


class Model_64(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc1 = nn.Linear(in_features=79200, out_features=256)
        self.fc21 = nn.Linear(in_features=256, out_features=10)
        self.fc22 = nn.Linear(in_features=256, out_features=10)

        self.fc3 = nn.Linear(in_features=10, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=79200)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=10, out_channels=10, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=10, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2, inplace=True),
        # )
        #
        # self.fc11 = nn.Linear(128 * 7 * 7, 32)
        # self.fc12 = nn.Linear(128 * 7 * 7, 32)
        # self.fc2 = nn.Linear(32, 128 * 7 * 7)
        #
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(inplace=True),
        #
        #     nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
        #     nn.Sigmoid()
        # )

    # 这个问题？？还是LOSS问题
    def reparameterize(self, mu, logvar):
        eps = torch.randn(mu.size(0), mu.size(1)).cuda()
        z = mu + eps * torch.exp(logvar / 2)

        # eps = torch.randn(mu.size()).cuda()
        # z = mu + eps * torch.exp(logvar/2)
        return z

    def forward(self, data):
        out = self.encoder(data)
        shape = out.shape
        out = out.flatten(start_dim=1)
        out1, out2 = self.fc1(out), self.fc1(out)
        mu, logvar = self.fc21(out1), self.fc22(out2)
        out = self.reparameterize(mu, logvar)
        out = self.fc3(out)
        out = self.fc4(out)
        out = out.reshape(shape)
        out = self.decoder(out)

        return out, mu, logvar
        #
        # out1, out2 = self.encoder(data), self.encoder(data)  # batch_s, 8, 7, 7
        # mu = self.fc11(out1.view(out1.size(0), -1))  # batch_s, latent
        # logvar = self.fc12(out2.view(out2.size(0), -1))  # batch_s, latent
        # z = self.reparameterize(mu, logvar)  # batch_s, latent
        # out3 = self.fc2(z).view(z.size(0), 128, 7, 7)  # batch_s, 8, 7, 7
        #
        # return self.decoder(out3), mu, logvar

class Model_128(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, stride=2, padding=(1, 0)),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc1 = nn.Linear(in_features=71300, out_features=256)

        self.fc21 = nn.Linear(in_features=256, out_features=10)
        self.fc22 = nn.Linear(in_features=256, out_features=10)

        self.fc3 = nn.Linear(in_features=10, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=71300)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=10, out_channels=10, kernel_size=5, stride=2, padding=(1, 0), output_padding=(1, 0)),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=10, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2, inplace=True),
        # )
        #
        # self.fc11 = nn.Linear(128 * 7 * 7, 32)
        # self.fc12 = nn.Linear(128 * 7 * 7, 32)
        # self.fc2 = nn.Linear(32, 128 * 7 * 7)
        #
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(inplace=True),
        #
        #     nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
        #     nn.Sigmoid()
        # )

    # 这个问题？？还是LOSS问题
    def reparameterize(self, mu, logvar):
        eps = torch.randn(mu.size(0), mu.size(1)).cuda()
        z = mu + eps * torch.exp(logvar / 2)

        # eps = torch.randn(mu.size()).cuda()
        # z = mu + eps * torch.exp(logvar/2)
        return z

    def forward(self, data):
        out = self.encoder(data)
        shape = out.shape
        out = out.flatten(start_dim=1)
        out1, out2 = self.fc1(out), self.fc1(out)
        mu, logvar = self.fc21(out1), self.fc22(out2)
        out = self.reparameterize(mu, logvar)
        out = self.fc3(out)
        out = self.fc4(out)
        out = out.reshape(shape)
        out = self.decoder(out)
        return out, mu, logvar


if __name__ == '__main__':
    model = Model_64().cuda()
    data = torch.ones(1, 1, 1980, 64).cuda()
    o, mu, logvar = model(data)
    print(o.shape, mu.shape, logvar.shape)