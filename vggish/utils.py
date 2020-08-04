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
def extractfeature(file_path):
#     wav, _ = librosa.load(file_path, sr=6000)
#     wav, _ = librosa.effects.trim(wav, top_db=20)
#     L = 6000 * 10
#     wav = wav[:L]
#     if wav.shape[0]<L:
#         padding_len = L - wav.shape[0]
#         wav = np.pad(wav,(int(padding_len/2), int(padding_len/2+0.5)), mode='constant')
#     wav = librosa.util.frame(wav, frame_length=8000, hop_length=1600)
#     win = librosa.filters.get_window('hanns')
#     wav = np.transpose(wav, (1, 0))
#     y = wav*win
#     librosa.core.fft_frequencies()
#     l = []
#     for i in range(10):
#         d = wav[i*16000:(i+1)*16000]
#         y = librosa.stft(d, n_fft=1200, hop_length=168, win_length=1200, window='hann')
#         magnitudes = np.abs(y) ** 2
#         mel = np.log10(np.dot(mel_basis, magnitudes) + 1e-6)
#         # print(mel.shape)
#         l.append(np.expand_dims(mel, axis=0))
#     mel = np.concatenate(l, axis=0)
#     torch.save(torch.from_numpy(mel), osp.join('../../data/vggish_pt/eval', osp.basename(file_path)))
    mel = torch.load(file_path)
    return mel


class GetData(Dataset):
    def  __init__(self, paths, patients_disease_dict):
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
        # if not isinstance(mfccs, np.ndarray):
        #     return []
        patient_id = file_path.split('/')[-1].split('_')[0]
        disease = self.patients_disease_dict[patient_id]

        # print(mfccs.shape, disease)
        return mfccs, disease

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
    train_file_paths = glob.glob(osp.join(data_path, 'train/*'))
    eval_file_paths = glob.glob(osp.join(data_path, 'eval/*'))
    train_set = GetData(train_file_paths, patients_disease_dict)
    eval_set = GetData(eval_file_paths, patients_disease_dict)
    train_loader = DataLoader(train_set, num_workers=20, shuffle=True, batch_size=batch_size)
    eval_loader = DataLoader(eval_set, num_workers=4, shuffle=False, batch_size=batch_size)
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
    balance_data(glob.glob('../../data/vggish_pt/eval/*'), patients_disease_dict)
