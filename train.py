import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import dataloader, log
import argparse
from tensorboardX import SummaryWriter


class Layer1(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru1 = nn.GRU(input_size=40, hidden_size=32, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(input_size=32, hidden_size=128, num_layers=1, batch_first=True)

        self.gru3 = nn.GRU(input_size=40, hidden_size=64, num_layers=1, batch_first=True)
        self.gru4 = nn.GRU(input_size=64, hidden_size=128, num_layers=1, batch_first=True)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, data):
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
        self.dense1 = nn.Linear(in_features=391*32, out_features=256*64)
        self.dense2 = nn.Linear(in_features=256*64, out_features=128*16)

        self.dense3 = nn.Linear(in_features=391*32, out_features=256*32)
        self.dense4 = nn.Linear(in_features=256*32, out_features=128*16)

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

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Layer1()
        self.layer2 = Layer2()
        self.layer3 = Layer3()

        self.dense1 = nn.Linear(in_features=128*16, out_features=64*16)
        self.dense2 = nn.Linear(in_features=64*16, out_features=8)

    def forward(self, data):
        temp, out = self.layer1(data)
        out = self.layer2(out)
        out = out + temp
        out = out.flatten(start_dim=1)
        out = self.layer3(out)
        out = self.dense1(out)
        out = F.leaky_relu(out, negative_slope=1e-2)
        out = F.dropout(out, p=0.5)
        out = self.dense2(out)
        return out

class Manager():
    def __init__(self, train_loader, eval_loader, model, learning_rate, save_model_path):
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.model = model
        self.save_model_path = save_model_path
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.train_logger, self.eval_logger = log()
        self.summarywriter_train = SummaryWriter(comment='-baseline-train')
        self.summarywriter_eval = SummaryWriter(comment='-baseline-train')
        self.disease_index = {'Healthy': 0, 'URTI': 1, 'LRTI': 2, 'Asthma': 3, 'COPD': 4,
                         'Bronchiectasis': 5, 'Bronchiolitis': 6, 'Pneumonia': 7}

    def fit(self):
        self.model.train()
        for epoch in range(1000):
            for step, (mfccs, labels) in enumerate(self.train_loader):
                mfccs = mfccs.permute(0, 2, 1).cuda()
                labels = labels.cuda()
                out = self.model(mfccs)
                loss = self.criterion(out, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if step % 3 ==0:
                    recall, precision, F1_score = self.indicators(out=out, labels=labels)
                    for r, p, f, diagnosis in zip(recall, precision, F1_score, self.disease_index.keys()):
                        self.summarywriter_train.add_scalars('indicators', {'F1_score_'+diagnosis:f.item(),
                                                              'Recall_'+diagnosis: r.item(),
                                                              'Precision_'+diagnosis:p.item()}, step)
                    self.summarywriter_train.add_scalar('loss', loss.item(), step)

                    self.train_logger.info('epoch:{},step:{}|{} loss:{}'
                                           .format(epoch, step, len(self.train_loader), loss))
            self.eval(epoch)

    def eval(self, epoch):
        self.model.eval()
        out = []
        label_list = []
        with torch.no_grad():
            for i, (mfccs, labels) in enumerate(self.eval_loader):
                mfccs = mfccs.permute(0, 2, 1).cuda()
                out.append(self.model(mfccs).cpu())
                label_list.append(labels)
            out = torch.cat(out, dim=0)
            labels = torch.cat(label_list, dim=0)
            loss = self.criterion(out, labels)
            recall, precision, F1_score = self.indicators(out=out, labels=labels)
            for r, p, f, diagnosis in zip(recall, precision, F1_score, self.disease_index.keys()):
                self.summarywriter_eval.add_scalars('indicators', {'F1_score_'+diagnosis:f.item(),
                                                              'Recall_'+diagnosis: r.item(),
                                                              'Precision_'+diagnosis:p.item()}, epoch)
            self.summarywriter_eval.add_scalar('loss', loss.item(), epoch)
            self.eval_logger.info('epoch:{}'.format(epoch))
            self.eval_logger.info('Label, Recall, Precision, F1_score'.format(epoch))
            for r, p, f, disease in zip(recall, precision, F1_score, self.disease_index.keys()):
                self.eval_logger.info('{}, {}, {}, {}'.format(disease, r, p, f))
        self.model.train()

    def indicators(self, out, labels):
        predict = out.argmax(dim=1)
        confusion_matrix = torch.zeros(8, 8)
        for p, t in zip(predict, labels):
            confusion_matrix[p.item(), t.item()] +=1
        # 要注意避免除以负数
        recall = confusion_matrix.diag() / (confusion_matrix.sum(0)+1e-16)
        precision = confusion_matrix.diag() / (confusion_matrix.sum(1)+1e-16)
        F1_score = 2*recall*precision/(recall+precision+1e-16)

        return recall, precision, F1_score

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.save_model_path, epoch + '.pth'))

    def load_model_state_dict(self, load_model_path):
        state_dict = self.model.state_dict()
        pretrain_state_dict = torch.load(load_model_path)
        state_dict.update(pretrain_state_dict)
        self.model.load_state_dict(state_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save-model-path', type=str, required=True)
    parser.add_argument('--load-model-path', type=str)
    parser.add_argument('--is-train', action='store_true', default=False)
    parser.add_argument('--batch-size', type=int, default=20)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    train_loader, eval_loader = dataloader(args.batch_size, '../data', '../data/diagnosis.txt')
    model = Model()
    model = nn.DataParallel(model.cuda())
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    manager = Manager(train_loader, eval_loader, model, args.lr, args.save_model_path)
    if args.load_model_path:
        manager.load_model_state_dict(args.load_model_path)
    if args.is_train:
        manager.fit()

