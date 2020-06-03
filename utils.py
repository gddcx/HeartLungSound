import librosa
import glob
import numpy as np
import os.path as osp
from torch.utils.data import Dataset, DataLoader
import logging.handlers
import datetime as dt

# 提取mfcc特征
def extractfeature(audio):
    """
    audio: 输入的一段音频
    :return 返回输入音频的mfcc特征
    """
    # sr=None会以原始频率读取音频，缺省则以22050Hz读取，设置其他则会上/下采样
    y, sr = librosa.load(audio, sr=None)
    y = librosa.to_mono(y)
    while y.shape[0] < 200000:
        y = np.concatenate((y, y), axis=0)
    y = y[:200000]
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return mfccs

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
        patient_id = file_path.split('/')[-1].split('_')[0]
        disease = self.patients_disease_dict[patient_id]
        return mfccs, disease

# 创建dataloader
def dataloader(batch_size, data_path, annotation_path):
    """
    path: 数据路径
    :rtype: dataloader
    """
    disease_index = {'Healthy':0, 'URTI':1, 'LRTI':2, 'Asthma':3, 'COPD':4,
                     'Bronchiectasis':5, 'Bronchiolitis':6, 'Pneumonia':7}
    patients_disease_dict = {}
    with open(annotation_path) as f:
        for l in f.readlines():
            lines_list = l.strip().split()
            patients_disease_dict[lines_list[0]] = disease_index[lines_list[1]]
    if len(patients_disease_dict) !=126:
        print('length of patients_disease_dict is not equal to 126!')
        exit(0)
    train_file_paths = glob.glob(osp.join(data_path, 'train/*'))
    eval_file_paths = glob.glob(osp.join(data_path, 'eval/*'))
    train_set = GetData(train_file_paths, patients_disease_dict)
    eval_set = GetData(eval_file_paths, patients_disease_dict)
    train_loader = DataLoader(train_set, num_workers=4, shuffle=True, batch_size=batch_size)
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


def balance_data(paths, patients_disease_dict):
    disease_index = {'Healthy': 0, 'URTI': 0, 'LRTI': 0, 'Asthma': 0, 'COPD': 0,
                          'Bronchiectasis': 0, 'Bronchiolitis': 0, 'Pneumonia': 0}
    for p in paths:
        id = p.split('/')[-1].split('_')[0]
        disease_index[patients_disease_dict[id]]+=1
    print(disease_index)

if __name__ == '__main__':
    patients_disease_dict = {}
    with open('../data/diagnosis.txt') as f:
        for l in f.readlines():
            lines_list = l.strip().split()
            patients_disease_dict[lines_list[0]] = lines_list[1]
    balance_data(glob.glob('../data/ICBHI_final_database/*.wav'), patients_disease_dict)
