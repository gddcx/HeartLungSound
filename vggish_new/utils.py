import librosa
import glob
import numpy as np
import os.path as osp
from torch.utils.data import Dataset, DataLoader
import logging.handlers
import datetime as dt
import scipy.io as scio
import torch

# 提取mel特征
mel_basis = librosa.filters.mel(sr=6000, n_fft=1200, n_mels=64)
def get_mel(y):
    y = librosa.core.stft(y=y, n_fft=1200,
                          hop_length=62,
                          win_length=400,
                          window='hann')
    magnitudes = np.abs(y) ** 2
    mel = np.log10(np.dot(mel_basis, magnitudes) + 1e-6)
    return mel

def extractfeature(file_path):
    wav, _ = librosa.load(file_path, sr=6000)
    wav, _ = librosa.effects.trim(wav, top_db=20)
    L = 6000 * 10
    wav = wav[:L]
    if wav.shape[0] < L:
        return
    mel = get_mel(wav)
    mel = mel[:, :960]
    return mel

class GetData(Dataset):
    def __init__(self, paths, patients_disease_dict):
        """
        paths: 音频文件路径
        patients_disease_dict: 病人ID和疾病对应字典，key是病人ID，value疾病类型
        """
        self.paths = paths
        self.patients_disease_dict = patients_disease_dict

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        file_path = self.paths[index]

        mfccs = extractfeature(file_path)
        if not isinstance(mfccs, np.ndarray):
            return False
        # if not isinstance(mfccs, np.ndarray):
        #     return []
        patient_id = file_path.split('/')[-1].split('_')[0]
        disease = self.patients_disease_dict[patient_id]

        # print(mfccs.shape, disease)
        return mfccs, disease
#
def my_collate(batchs):
    data_list = []
    label_list = []
    for b in batchs:
        if b:
            data, label = b
            data = np.expand_dims(data, axis=0)
            data_list.append(data)
            label_list.append(label)
    data = np.concatenate(data_list, axis=0)
    label = np.asarray(label_list)
    return torch.from_numpy(data), torch.from_numpy(label)

# 创建dataloader
def dataloader(batch_size, data_path, annotation_path):
    """
    path: 数据路径
    :rtype: dataloader
    """
    disease_index = {'Healthy':0, 'URTI':1, 'COPD':2,
                     'Bronchiectasis':3, 'Bronchiolitis':4, 'Pneumonia':5}
    patients_disease_dict = {}
    with open(annotation_path) as f:
        for l in f.readlines():
            lines_list = l.strip().split()
            try:
                patients_disease_dict[lines_list[0]] = disease_index[lines_list[1]]
            except KeyError:
                print(lines_list[1])
    train_file_paths = glob.glob(osp.join(data_path, 'train/*-norm.wav'))
    eval_file_paths = glob.glob(osp.join(data_path, 'eval/*-norm.wav'))
    train_set = GetData(train_file_paths, patients_disease_dict)
    eval_set = GetData(eval_file_paths, patients_disease_dict)
    train_loader = DataLoader(train_set, num_workers=20, shuffle=True, batch_size=batch_size, collate_fn=my_collate)
    eval_loader = DataLoader(eval_set, num_workers=4, shuffle=False, batch_size=batch_size, collate_fn=my_collate)
    return train_loader, eval_loader

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


# 统计各类疾病的文件数量
pneumonia_list = []
def balance_data(paths, patients_disease_dict):
    disease_index = {'Healthy': 0, 'URTI': 0, 'LRTI': 0, 'Asthma': 0, 'COPD': 0,
                          'Bronchiectasis': 0, 'Bronchiolitis': 0, 'Pneumonia': 0}
    for p in paths:
        id = p.split('/')[-1].split('_')[0]
        # 统计某一类的ID
        if patients_disease_dict[id]=='COPD':
            pneumonia_list.append(id)
        disease_index[patients_disease_dict[id]]+=1
    pneumonia_list.sort()
    pneumonia_set = set(pneumonia_list)
    print(disease_index)
    print(pneumonia_set)

if __name__ == '__main__':
    patients_disease_dict = {}
    with open('../../data/diagnosis.txt') as f:
        for l in f.readlines():
            lines_list = l.strip().split()
            patients_disease_dict[lines_list[0]] = lines_list[1]
    balance_data(glob.glob('../../data/vggish_pt/train/*norm.wav'), patients_disease_dict)
