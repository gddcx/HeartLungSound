import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import argparse
from network import *
import logging.handlers
import datetime as dt
import torch.nn.functional as F
from PIL import Image
import glob


torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

def log():
    logger = {
        "train": logging.getLogger('train_log'),
        "dev": logging.getLogger('dev_log')
    }
    logger["train"].setLevel(logging.DEBUG)
    logger["dev"].setLevel(logging.DEBUG)
    format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler = {
        "train": logging.handlers.TimedRotatingFileHandler('./train.log', when='midnight', interval=1,
                                                           backupCount=7, atTime=dt.time(0, 0, 0, 0)),
        "dev": logging.handlers.TimedRotatingFileHandler('./dev.log', when='midnight', interval=1,
                                                         backupCount=7, atTime=dt.time(0, 0, 0, 0))
    }
    handler["train"].setFormatter(format)
    handler["dev"].setFormatter(format)
    logger["train"].addHandler(handler["train"])
    logger["dev"].addHandler(handler["dev"])

    return logger['train'], logger['dev']

class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 用BCELoss有问题
        # self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
    # 这个问题吗？？
    def forward(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # print('-'*50)
        # print('calculate loss')
        # print('-' * 50)
        # print(recon_x.shape, x.shape)
        # BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
        # BCE = self.bce_loss(recon_x, x)
        # print('-' * 50)
        # print('BCE:', BCE)
        # print('-' * 50)
        # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # print('-' * 50)
        # print('KLD:', KLD)
        # print('-' * 50)

        return BCE+KLD

class Manager():
    def __init__(self, args, model, criterion, train_loader, eval_loader, logger):
        self.epoch = 0
        self.args = args
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.logger = logger
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr,
                                          betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    def train(self):
        self.model.train()
        total_epoch = 10000
        for epoch in range(self.epoch, total_epoch):
            for step, data in enumerate(self.train_loader):
                data = data.cuda()
                out, mu, logvar = self.model(data)
                loss = self.criterion(out, data, mu, logvar)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if step % 50 == 0:
                    self.logger[0].info('Epoch:{}, Step:{}/{}, Loss:{}'.
                                              format(epoch, step, len(self.train_loader), loss.data))

            self.eval(epoch=epoch)
            if epoch % 200 == 0:
                self.save_model(epoch)
        self.save_model(total_epoch)


    def eval(self, epoch):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for step, data in enumerate(self.eval_loader):
                data = data.cuda()
                out, mu, logvar = self.model(data)
                loss = self.criterion(out, data, mu, logvar)
                total_loss += loss.data
            self.logger[1].info('Epoch:{}, Loss{}'.format(epoch, total_loss))
        self.model.train()

    def test(self):
        with torch.no_grad():
            import matplotlib.pyplot as plt
            for step, data in enumerate(self.eval_loader):
                data = data.cuda()
                out, _, _ = self.model(data)
                out = out.squeeze(dim=0)

                # plt.imsave('result1.png', out.cpu().permute(1, 2, 0))
                # plt.imsave('raw.png', data.squeeze(dim=0).cpu().permute(1, 2, 0))
                torchvision.utils.save_image(out, 'result1.png')
                data = data.squeeze(dim=0)
                torchvision.utils.save_image(data, 'raw.png')
                return


    def load_state_dict(self, path):
        print('Load pre-train state dict from', path)
        checkpoint = torch.load(path)
        # print(checkpoint)
        state_dict = checkpoint['state_dict']
        self.epoch = checkpoint['epoch']
        optimizer = checkpoint['optimizer']
        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(optimizer)


    def save_model(self, epoch):
        torch.save({'epoch': epoch, 'state_dict':self.model.state_dict(), 'optimizer': self.optimizer.state_dict()},
                   './state_dict'+str(epoch)+'ICBHI_128.pth')

class GetData(torch.utils.data.Dataset):
    def __init__(self, path, compose):
        super().__init__()
        self.path = glob.glob(path)
        self.compose = compose

    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):
        path = self.path[index]
        img = Image.open(path).convert('L')
        img = self.compose(img)
        return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--load-model-path', type=str)
    parser.add_argument('--save-model-path', type=str, default='.')
    parser.add_argument('--is-train', action='store_true', default=False)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

    logger = log()

    compose_64 = torchvision.transforms.Compose([
        torchvision.transforms.Resize((64, 1980)),
        torchvision.transforms.ToTensor()
    ])
    compose_128 = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 926)),
        torchvision.transforms.ToTensor()
    ])
    # train_set = torchvision.datasets.MNIST(root='../../data/MNIST/train', train=True, download=True, transform=compose_128)
    # test_set = torchvision.datasets.MNIST(root='../../data/MNIST/test', train=False, download=True, transform=compose_128)
    # train_set = GetData(path='../../data/VAE_64/train/*.png', compose= compose_64)
    # test_set = GetData(path='../../data/VAE_64/eval/*.png', compose= compose_64)
    train_set = GetData(path='../../data/VAE_128/train/*.png', compose=compose_128)
    test_set = GetData(path='../../data/VAE_128/eval/*.png', compose=compose_128)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=12, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=12, shuffle=False)

    model = Model_128()
    model = nn.DataParallel(model.cuda())

    loss = VAELoss()
    manger = Manager(args, model, loss, train_loader, test_loader, logger)
    # state_dict1000MNIST_128较好
    if args.load_model_path:
        manger.load_state_dict(args.load_model_path)
    if args.is_train:
        manger.train()
    else:
        manger.test()
